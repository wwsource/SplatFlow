# @File: train_kitti.sh
# @Project: SplatFlow
# @Author : wangbo
# @Time : 2024.07.03

export CUDA_VISIBLE_DEVICES="0, 1, 2"
python -m torch.distributed.launch --nproc_per_node 3 run_train.py \
--exp_name train_kitti \
--stage kitti \
--pre_name_path exp/train_sintel/model.pth \
--image_size 368 768 \
--batch_size 6 \
--lr 0.000125 \
--wdecay 0.00001 \
--step_max 50000 \
--log_train 100 \