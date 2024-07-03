export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"
python -m torch.distributed.launch --nproc_per_node 8 run_train.py \
--exp_name train_things_part \
--part_params_train \
--stage things \
--pre_name_path exp/0-pretrain/gma-things.pth \
--image_size 400 720 \
--batch_size 8 \
--lr 0.000125 \
--wdecay 0.0001 \
--step_max 100000 \
--log_train 100 \

python -m torch.distributed.launch --nproc_per_node 8 run_train.py \
--exp_name train_things_full \
--stage things \
--pre_name_path exp/train_things_part/model.pth \
--image_size 400 720 \
--batch_size 8 \
--lr 0.000125 \
--wdecay 0.00001 \
--step_max 100000 \
--log_train 100 \
