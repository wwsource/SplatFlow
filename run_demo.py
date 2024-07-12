# @File: run_demo.py
# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

import os

import torch
from model.model_splatflow import SplatFlow
from model.util.util import *

print('SplatFlow demo start...')

model = SplatFlow()
model.load_state_dict(torch.load('exp/0-pretrain/splatflow-kitti-50000.pth'), strict=True)
model.eval().cuda()
print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

img_names = ['data/demo/image/000009_%02d.png'%i for i in [9, 10, 11]]

imgs = [torch.from_numpy(np.array(readImageKITTI(img_names[i])).astype(np.uint8)).permute(2, 0, 1).float()[None].cuda() for i in range(3)]
img0, img1, img2 = imgs
padder = InputPadder(img1.shape)
img0, img1, img2 = padder.pad(img0, img1, img2)

with torch.no_grad():
    outputs = model.infer(
        model,
        input_list=[img0, img1, img2],
        iters=24)

    pr_flow2d = padder.unpad(outputs[0])[0][0].permute(1, 2, 0).cpu().numpy()

    output_path = 'exp/demo'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writeFlowKITTI(f'{output_path}/flow_000009_10.png', pr_flow2d)

print('Success!!!')
