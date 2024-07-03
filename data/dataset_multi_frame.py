# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

import os.path as osp
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.util import util
from model.util.augmentor import FlowAugmentor, SparseFlowAugmentor

data_root = '/data1/wangbo/data/'
things_root = data_root + 'FlyingThings3D'
kitti_root = data_root + 'KITTI'

class FlowDataset(Dataset):
    def __init__(self, aug_params=None, sparse=False):

        self.sparse = sparse
        self.aug_params = None
        self.augmentor = None
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.image_list = []
        self.flow_list = []
        self.occ_list = []

    def __getitem__(self, index):

        index = index % len(self.image_list)

        img1_dir = self.image_list[index][0]
        img2_dir = self.image_list[index][1]
        img3_dir = self.image_list[index][2]

        img1 = util.read_gen(img1_dir)
        img2 = util.read_gen(img2_dir)
        img3 = util.read_gen(img3_dir)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)

        flow1_dir = self.flow_list[index][0]
        flow2_dir = self.flow_list[index][1]

        if flow2_dir is not None:
            need_init = 0
            valid1 = None
            valid2 = None
            if self.sparse:
                flow1, valid1 = flow2, valid2 = util.readFlowKITTI(flow2_dir)
                if flow1_dir is not None:
                    flow1, valid1 = util.readFlowKITTI(flow1_dir)
                    need_init = 1
            else:
                flow1 = flow2 = util.read_gen(flow2_dir)
                if flow1_dir is not None:
                    flow1 = util.read_gen(flow1_dir)
                    need_init = 1

            flow1 = np.array(flow1).astype(np.float32)
            flow2 = np.array(flow2).astype(np.float32)

            if len(img2.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))
                img3 = np.tile(img3[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]
                img2 = img2[..., :3]
                img3 = img3[..., :3]

            if self.augmentor is not None:
                if self.sparse:
                    img1, img2, img3, flow1, valid1, flow2, valid2 = self.augmentor(img1, img2, img3, flow1, valid1,
                                                                                    flow2, valid2)
                else:
                    img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
            flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
            flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

            if valid1 is not None:
                valid1 = torch.from_numpy(valid1)
            else:
                valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)

            if need_init == 0:
                valid1 = torch.zeros_like(valid1).bool()

            if valid2 is not None:
                valid2 = torch.from_numpy(valid2)
            else:
                valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)

            valid1 = valid1.float()
            valid2 = valid2.float()

            return img1, img2, img3, flow1, valid1, flow2, valid2

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        self.flow_list = v * self.flow_list
        return self

    def __len__(self):
        return len(self.image_list)

class Things(FlowDataset):
    def __init__(self, aug_params=None, root=things_root, split='train', ptype='clean'):
        super(Things, self).__init__(aug_params)

        split_dir = {'train': 'TRAIN', 'val': 'TEST'}[split]
        ptype_dir = {'clean': 'frames_cleanpass', 'final': 'frames_finalpass'}[ptype]

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_paths = sorted(glob(osp.join(root, ptype_dir, split_dir, '*/*')))
                image_paths = sorted([osp.join(f, cam) for f in image_paths])

                flow_paths = sorted(glob(osp.join(root, 'optical_flow', split_dir, '*/*')))
                flow_paths = sorted([osp.join(f, direction, cam) for f in flow_paths])

                for ipath, fpath in zip(image_paths, flow_paths):
                    images = sorted(glob(osp.join(ipath, '*.png')))
                    flows = sorted(glob(osp.join(fpath, '*.pfm')))

                    images = [img.replace('\\', '/') for img in images]
                    flows = [flow.replace('\\', '/') for flow in flows]

                    if split == 'train':
                        for i in range(len(flows) - 2):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1], images[i + 2]]]
                                self.flow_list += [[flows[i], flows[i + 1]]]

                            elif direction == 'into_past':
                                self.image_list += [[images[i + 2], images[i + 1], images[i]]]
                                self.flow_list += [[flows[i + 2], flows[i + 1]]]

                    elif split == 'val':
                        if direction == 'into_future':
                            self.image_list += [[images[3], images[4], images[5]]]
                            self.flow_list += [[None, flows[4]]]

                        elif direction == 'into_past':
                            self.image_list += [[images[6], images[5], images[4]]]
                            self.flow_list += [[None, flows[5]]]

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, root=kitti_root, split='train'):
        super(KITTI, self).__init__(aug_params, sparse=True)

        split_dir = {'train': 'training', 'test': 'testing'}[split]

        root = osp.join(root, split_dir)

        imgs1 = sorted(glob(osp.join(root, 'image_2_multiview/*_09.png')))
        imgs2 = sorted(glob(osp.join(root, 'image_2_multiview/*_10.png')))
        imgs3 = sorted(glob(osp.join(root, 'image_2_multiview/*_11.png')))

        imgs1 = [img.replace('\\', '/') for img in imgs1]
        imgs2 = [img.replace('\\', '/') for img in imgs2]
        imgs3 = [img.replace('\\', '/') for img in imgs3]

        for img1, img2, img3 in zip(imgs1, imgs2, imgs3):
            self.image_list += [[img1, img2, img3]]

        if split == 'train':
            flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            flow_list = [flow.replace('\\', '/') for flow in flow_list]
            self.flow_list += [[None, flow] for flow in flow_list]
        elif split == 'test':
            self.flow_list += [[None, None] for _ in range(len(self.image_list))]

from contextlib import contextmanager
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def fetch_dataloader(config):
    def prepare_data(config):

        if config.stage == 'things':
            aug_params = {'crop_size': config.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
            things_clean = Things(aug_params, split='train', ptype='clean')
            things_final = Things(aug_params, split='train', ptype='final')
            dataset = things_clean + things_final

        return dataset

    if config.is_ddp:
        with torch_distributed_zero_first(config.rank):
            dataset = prepare_data(config)
    else:
        dataset = prepare_data(config)

    batch_size_tmp = config.batch_size // config.world_size

    dataloder = DataLoader(dataset,
                           batch_size=batch_size_tmp,
                           pin_memory=True,
                           sampler=torch.utils.data.distributed.DistributedSampler(dataset) if config.is_ddp else None,
                           num_workers=8 if config.is_ddp else 0,
                           drop_last=True)

    if config.is_master:
        print('Training with %d image pairs' % len(dataset))

    return dataloder

