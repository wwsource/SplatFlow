# @File: run_test.py
# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

import argparse
import torch.nn.functional as F

from model.model_splatflow import SplatFlow
from data.dataset import *

def get_stamp(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return '{}/{}/{}'.format(int(d), int(h), int(m))

class InputPadder:

    def __init__(self, dims, mode='sintel', base=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // base) + 1) * base - self.ht) % base
        pad_wd = (((self.wd // base) + 1) * base - self.wd) % base
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

@torch.no_grad()
def validate_things(model):

    print('Start testing splatflow on Things...')

    for ptype in ['clean', 'final']:

        epe_list = []

        val_dataset = Things(split='val', ptype=ptype)
        data_num = len(val_dataset)
        print(f'Dataset length {data_num}')

        for val_id in range(data_num):

            img1, img2, img3, gt1, valid1, gt2, valid2 = val_dataset[val_id]

            img1 = img1[None].cuda()
            img2 = img2[None].cuda()
            img3 = img3[None].cuda()
            padder = InputPadder(img2.shape)
            img1, img2, img3 = padder.pad(img1, img2, img3)

            flow_prs_23 = model.infer(
                model,
                input_list=[img1, img2, img3],
                iters=24)
            pr2 = padder.unpad(flow_prs_23[-1][0]).cpu()

            epe = torch.sum((pr2 - gt2) ** 2, dim=0).sqrt()
            epe = epe.view(-1)
            val2 = valid2.view(-1) >= 0.5
            epe_list.append(epe[val2].numpy())

            if val_id % 50 == 0:
                print(f'{ptype}: {val_id}/{data_num}')

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)

        print("Things (%s) EPE: %f" % (ptype, epe))

@torch.no_grad()
def validate_kitti(model):

    out_list, epe_list = [], []
    val_dataset = KITTI(split='train')
    data_num = len(val_dataset)
    print(f'Dataset length {data_num}')

    for val_id in range(data_num):
        img1, img2, img3, gt1, valid1, gt2, valid2 = val_dataset[val_id]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()
        img3 = img3[None].cuda()
        padder = InputPadder(img2.shape, mode='kitti')
        img1, img2, img3 = padder.pad(img1, img2, img3)

        flow_prs_23 = model.infer(
            model,
            input_list=[img1, img2, img3],
            iters=24)
        pr2 = padder.unpad(flow_prs_23[-1][0]).cpu()

        epe = torch.sum((pr2 - gt2) ** 2, dim=0).sqrt()
        mag = torch.sum(gt2 ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val2 = valid2.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val2].mean().item())
        out_list.append(out[val2].cpu().numpy())

        if val_id % 20 == 0:
            print(f'kitti: {val_id}/{data_num}')

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    fl = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, fl))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='things')
    parser.add_argument('--pre_name_path', default='exp/train_things_full/model.pth')

    args = parser.parse_args()

    model = SplatFlow()
    pre_replace_list = [
        ['update_block', 'update'], ['module.', ''],
        ['gru_tf', 'gru_sp'], ['flow_head_tf', 'flow_head_sp'], ['mask_tf', 'mask_sp']
    ]
    checkpoint = torch.load(args.pre_name_path)
    for l in pre_replace_list:
        checkpoint = {k.replace(l[0], l[1]): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)
    print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.eval().cuda()

    if args.dataset == 'things':
        validate_things(model)

    if args.dataset == 'kitti':
        validate_kitti(model)


