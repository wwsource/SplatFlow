# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import BasicEncoder
from .corr import CorrBlock
from .attention import Attention
from .softsplat import FunctionSoftsplat as forward_warping
from .update import Update
autocast = torch.cuda.amp.autocast
fast_inference = False
import torch.distributed as dist

class SplatFlow(nn.Module):
    def __init__(self, config=None):
        super(SplatFlow, self).__init__()

        self.hdim = self.cdim = 128
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=self.hdim + self.cdim, norm_fn='batch')
        self.att = Attention(dim=self.cdim, heads=1, dim_head=self.cdim)

        if config != None and config.part_params_train:
            for p in self.parameters():
                p.requires_grad = False

        self.update = Update(config, hidden_dim=self.hdim)

    def init_coord(self, fmap):
        f_shape = fmap.shape
        H, W = f_shape[-2:]
        y0, x0 = torch.meshgrid(
            torch.arange(H).to(fmap.device).float(),
            torch.arange(W).to(fmap.device).float())
        coord = torch.stack([x0, y0], dim=0)  # shape: (2, H, W)
        coord = coord.unsqueeze(0).repeat(f_shape[0], 1, 1, 1)
        return coord

    def initialize_flow(self, fmap):

        coords0 = self.init_coord(fmap)
        coords1 = self.init_coord(fmap)

        return coords0, coords1

    def cvx_upsample(self, data, mask):

        N, C, H, W = data.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(data, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, mf_t=None):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=fast_inference):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)

        coords0, coords1 = self.initialize_flow(fmap1)

        with autocast(enabled=fast_inference):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            atte_s = self.att(inp)

        flow_predictions = []

        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=fast_inference):
                net, up_mask, delta_flow, mf = self.update(net, inp, corr, flow, atte_s, mf_t)
                coords1 = coords1 + delta_flow

                if (fast_inference and (itr == iters - 1)) or (not fast_inference):

                    flow_up = self.cvx_upsample(8 * (coords1 - coords0), up_mask)
                    flow_predictions.append(flow_up)

            low = coords1 - coords0

        return flow_predictions, mf, low, fmap1, fmap2

    def Loss(self, flow_prs_01, gt_01, valid_01, flow_prs_12, gt_12, valid_12):

        MAX_FLOW = 400

        n_predictions = len(flow_prs_12)
        loss = 0

        valid_01 = ((valid_01 >= 0.5) & ((gt_01 ** 2).sum(dim=1).sqrt() < MAX_FLOW)).view(-1) >= 0.5
        valid_12 = ((valid_12 >= 0.5) & ((gt_12 ** 2).sum(dim=1).sqrt() < MAX_FLOW)).view(-1) >= 0.5

        for i in range(n_predictions):
            i_weight = 0.8 ** (n_predictions - i - 1)
            tmp_01 = ((flow_prs_01[i] - gt_01).abs().sum(dim=1)).view(-1)[valid_01]
            tmp_12 = ((flow_prs_12[i] - gt_12).abs().sum(dim=1)).view(-1)[valid_12]
            loss += i_weight * torch.cat([tmp_01, tmp_12]).mean()

        with torch.no_grad():
            epe = torch.sum((flow_prs_12[-1] - gt_12) ** 2, dim=1).sqrt()
            epe = epe.view(-1)[valid_12.view(-1)]
            epe_sum = epe.sum()
            px1_sum = (epe < 1).float().sum()
            px3_sum = (epe < 3).float().sum()
            px5_sum = (epe < 5).float().sum()
            valid_12_sum = valid_12.sum()

            dist.all_reduce(epe_sum)
            dist.all_reduce(px1_sum)
            dist.all_reduce(px3_sum)
            dist.all_reduce(px5_sum)
            dist.all_reduce(valid_12_sum)

            epe = epe_sum / valid_12_sum
            px1 = px1_sum / valid_12_sum
            px3 = px3_sum / valid_12_sum
            px5 = px5_sum / valid_12_sum

            metric_list = [
                ['epe', epe.item()],
                ['px1', px1.item()],
                ['px3', px3.item()],
                ['px5', px5.item()]]

        return loss, metric_list

    def infer(self, model, input_list, iters=12, gt_list=None, mf_01=None, low_01=None):

        img0, img1, img2 = input_list

        if img0 == None:
            flow_prs_12, mf_12, low_12, fmap1, fmap2 = model(img1, img2, iters=iters)
            return flow_prs_12

        if not (gt_list == None and mf_01 != None and low_01 != None):
            flow_prs_01, mf_01, low_01, fmap0, fmap1 = model(img0, img1, iters=iters)

        mf_t = forward_warping(mf_01, low_01)

        flow_prs_12, mf_12, low_12, fmap1, fmap2 = model(img1, img2, iters=iters, mf_t=mf_t)

        if gt_list != None:  # training mode
            gt_01, valid_01, gt_12, valid_12 = gt_list
            loss, metric_list = self.Loss(flow_prs_01, gt_01, valid_01, flow_prs_12, gt_12, valid_12)
            return loss, metric_list

        return flow_prs_12

    def training_infer(self, model, step_data, device):

        img0, img1, img2, gt_01, valid_01, gt_12, valid_12 = [x.to(device) for x in step_data]

        loss, metric_list = model.module.infer(
            model,
            input_list=[img0, img1, img2],
            gt_list=[gt_01, valid_01, gt_12, valid_12])

        return loss, metric_list

