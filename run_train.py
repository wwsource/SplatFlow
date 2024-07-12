# @File: run_train.py
# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

import argparse
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

from model.model_splatflow import SplatFlow
from data.dataset import *

def get_stamp(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return '{}/{}/{}'.format(int(d), int(h), int(m))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default='train_things_part')
    parser.add_argument('--stage', default='things')
    parser.add_argument('--pre_name_path', default='exp/0-pretrain/gma-things.pth')
    parser.add_argument('--part_params_train', action='store_true')
    parser.add_argument('--image_size', type=int, nargs='+', default=[400, 720])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.000125)
    parser.add_argument('--wdecay', type=float, default=.0001)
    parser.add_argument('--step_max', type=int, default=100000)
    parser.add_argument('--log_train', type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    args = parser.parse_args()

    args.rank = rank = args.local_rank
    args.is_master = is_master = True if rank in [0, -1] else False
    args.is_ddp = is_ddp = True if rank != -1 else False

    if is_ddp:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = world_size = dist.get_world_size()
    else:
        exit()

    model = SplatFlow(args).to(device)
    pre_replace_list = [
        ['update_block', 'update'], ['module.', ''],
        ['gru_tf', 'gru_sp'], ['flow_head_tf', 'flow_head_sp'], ['mask_tf', 'mask_sp']
    ]
    checkpoint = torch.load(args.pre_name_path, map_location=device)
    for l in pre_replace_list:
        checkpoint = {k.replace(l[0], l[1]): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=False)

    if is_master: print(f"Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device).train()
    model = DDP(model, device_ids=[rank], output_device=(rank))

    train_loader = fetch_dataloader(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.step_max + 100,
                                                pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    t_1 = t_0 = time.time()
    step_i = 1
    epoch = 1

    while epoch:
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        epoch = epoch + 1

        for step_data in train_loader:

            loss, metric_list = model.module.training_infer(model, step_data, device)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step_i % args.log_train == 0:
                if args.is_master:
                    t_now = time.time()
                    t_have = t_now - t_0
                    t_period = t_now - t_1
                    t_1 = t_now
                    t_left = t_period * (args.step_max - step_i) / args.log_train
                    time_stamp = 'time: [' + get_stamp(t_have) + ',' + get_stamp(t_left) + ']'
                    metric_log_list = [(mr[0] + ': %.3f') % mr[1] for mr in metric_list]
                    metric_log = '  '.join(metric_log_list)
                    print(f'{args.exp_name}\ttrain [{step_i}/{args.step_max}]\tloss: %.3f\t%s\tlr: %.6f\t' % (loss.item(), metric_log, scheduler.get_last_lr()[-1]) + time_stamp)

                dist.barrier()

            step_i += 1
            if step_i > args.step_max:
                if args.is_master:
                    model_path = f'exp/{args.exp_name}'
                    if not os.path.exists(model_path):
                        os.mkdir(model_path)
                    model_name_path = f'{model_path}/model.pth'
                    torch.save(model.state_dict(), model_name_path)
                    print('training finished')

                dist.barrier()

                exit()









