export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5"
python -m torch.distributed.launch --nproc_per_node 6 run_train.py \
--exp_name train_sintel \
--stage sintel \
--pre_name_path exp/train_things_full/model.pth \
--image_size 368 768 \
--batch_size 6 \
--lr 0.000125 \
--wdecay 0.00001 \
--step_max 120000 \
--log_train 100 \