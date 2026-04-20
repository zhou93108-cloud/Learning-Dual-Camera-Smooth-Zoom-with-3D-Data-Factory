# EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29516 train.py \
                                --cfg config/eval/kitti-S.json \
                                --model ../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-S.pth \
                                --log_dir ./xsmall/xiaomi06/small4_lr075_noload \
                                --dataset_dir_real ./ \
                                -dataset_dir_syn ./ \
                                -dataset_name xiaomi \
                                --epoch 100 \
                                --world_size=2

