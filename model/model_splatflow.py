import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import BasicEncoder
from .corr import CorrBlock
from .attention import Attention
from .softsplat import FunctionSoftsplat as forward_warping
from .update import Update
autocast = torch.cuda.amp.autocast
fast_inference = True

class SplatFlow(nn.Module):
    def __init__(self):
        super(SplatFlow, self).__init__()

        self.hdim = self.cdim = 128
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=self.hdim + self.cdim, norm_fn='batch')
        self.att = Attention(dim=self.cdim, heads=1, dim_head=self.cdim)
        self.update = Update(hidden_dim=self.hdim)

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

    def infer(self, model, input_list, iters=12):

        img0, img1, img2 = input_list

        flow_prs_01, mf_01, low_01, fmap0, fmap1 = model(img0, img1, iters=iters)

        mf_t = forward_warping(mf_01, low_01)

        flow_prs_12, mf_12, low_12, fmap1, fmap2 = model(img1, img2, iters=iters, mf_t=mf_t)

        return flow_prs_12[-1],
