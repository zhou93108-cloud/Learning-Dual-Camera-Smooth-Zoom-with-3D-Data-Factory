import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from datetime import timedelta

from LDCSZ_model import Model

from dataset_color_xiaomi import DZDataset as DZDataset_xiaomi
from dataset_color_huawei import DZDataset as DZDataset_huawei
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import DZDataset

from config.parser import parse_args

device = torch.device("cuda")

def crop_center(hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3.0e-4 * mul
    else:
        mul = np.cos((step - 2000.) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3.0e-4 - 3.0e-6) * mul + 3.0e-6
    
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, data_name, data_root, data_root_syn, log_dir, local_rank):
    step = 0
    if data_name == 'xiaomi':
        dataset = DZDataset_xiaomi('train', data_root=data_root, data_root_syn=data_root_syn)
    elif data_name == 'huawei':
        dataset = DZDataset_huawei('train', data_root=data_root, data_root_syn=data_root_syn)
    else:
        raise ValueError("Unrecognized data name")
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data

            b, t, c, h, w = data_gpu.shape
            data_gpu = data_gpu.view(-1, c, h, w)
            timestep = timestep.view(-1, timestep.shape[-3], timestep.shape[-2], timestep.shape[-1])

            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)

            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / (4 * 1)
            pred, info = model.update(imgs, gt, epoch, timestep=timestep, learning_rate=learning_rate, training=True) # pass timestep if you are training RIFEm
            #pred, info = model.update(imgs, gt, epoch, timestep=timestep, learning_rate=learning_rate, training=True) # pass timestep if you are training RIFEm

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e} loss_vgg:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1'],  info['loss_vgg']))
            step += 1
        model.save_model(log_dir, local_rank) 
        if epoch == 99:
            model.save_model(log_dir, local_rank, suffix=str(epoch)) 
        dist.barrier()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="finetune FT model")
    parser.add_argument('--cfg', default='config/eval/kitti-M.json', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', default='../SEA-RAFT-main/pretrained/Tartan-C-T-TSKH-kitti432x960-M.pth', help='checkpoint path', required=True, type=str)
    parser.add_argument("--log_dir", type=str, default="./ckpt/RIFE_finetuned", help="log path")
    parser.add_argument("--dataset_dir_real", type=str, default="../", help="Real ZoomGS train data path")
    parser.add_argument("--dataset_dir_syn", type=str, default="../", help="Syn ZoomGS train data path")
    parser.add_argument("--dataset_name", type=str, default="xiaomi", help="Use data from xiaomi or huawei")
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--local-rank', dest='local_rank', default=0, type=int, help='local rank')
    args = parse_args(parser)
    #args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size, timeout=timedelta(seconds=72000))

    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly = True

    model = Model(args, local_rank=args.local_rank)
    
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    data_root = args.dataset_dir_real
    data_root_syn = args.dataset_dir_syn
    data_name = args.dataset_name

    train(model, data_name, data_root, data_root_syn, log_dir, args.local_rank)
        


