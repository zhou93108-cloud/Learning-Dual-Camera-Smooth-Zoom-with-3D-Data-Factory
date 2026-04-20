import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as tra
import glob

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice 

class DZDataset(Dataset):
    def __init__(self, dataset_name, data_root, data_root_syn, batch_size=1):
        self.batch_size = batch_size
        self.dataset_name = dataset_name        
        self.data_root = data_root
        self.data_root_syn = data_root_syn

        self.train_cache = {}

        train_ids = os.listdir(os.path.join(self.data_root, "train"))
        for id in train_ids:
            self.train_cache[id] = []
            files = sorted(glob.glob(os.path.join(self.data_root, "train", id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
            
            for file in files:
                self.train_cache[id].append(file)
        
        aditional_ids = os.listdir(self.data_root_syn)
        for aditional_id in aditional_ids:
            dl_ids = os.listdir(os.path.join(self.data_root_syn, aditional_id))
            for dl_id in dl_ids:
                self.train_cache[dl_id] = []
                files = sorted(glob.glob(os.path.join(self.data_root_syn, aditional_id, dl_id, '*.png')), key=lambda f:int(f.split('/')[-1].split('.')[0].split('_')[-1]))
            
                for file in files:
                    self.train_cache[dl_id].append(file)


        self.load_data()
        print(len(list(self.meta_data.keys())))
    def __len__(self):
        return 40000 * 1
        
    def load_data(self):
        self.meta_data = self.train_cache

    def crop_center(self, hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w, :]
    
    def crop_center_gray(self, hr, ph, pw):
        ih, iw = hr.shape[0:2]
        lr_patch_h, lr_patch_w = ph, pw
        ph = ih // 2 - lr_patch_h // 2
        pw = iw // 2 - lr_patch_w // 2
        return hr[ph:ph+lr_patch_h, pw:pw+lr_patch_w]

    def __getitem__(self, index): 
        index = index % len(self.meta_data.keys())

        img0s = []
        gts = []
        img1s = []
        timesteps = []
        
        img0 = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][0])
        img0 = cv2.resize(img0, (1216,1632), interpolation=cv2.INTER_LINEAR)
        img0_c = img0

        # W Image
        #img1_c = self.meta_data[list(self.meta_data.keys())[index]][-1]
        img1_c = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][-1])
        img1_c = cv2.resize(img1_c, (1216,1632), interpolation=cv2.INTER_LINEAR)

        # UW -> W transition image
        ii = random.randint(1,  32 - 2)   ######## change ########
        timestep = ii / 31       ######## change ########
        t = ii / 31
        #gt = self.meta_data[list(self.meta_data.keys())[index]][ii]
        gt = cv2.imread(self.meta_data[list(self.meta_data.keys())[index]][ii])
        gt = cv2.resize(gt, (1216,1632), interpolation=cv2.INTER_LINEAR)
        gt_c = gt
        img0s.append(img0_c)
        img1s.append(img1_c)
        gts.append(gt_c)
        #print(img0_gray.shape)
        timesteps.append(timestep)
       
        img0s = np.array(img0s)
        img1s = np.array(img1s)
        gts = np.array(gts)
        
        # 正态分布取值
        brightnesses = np.random.normal(1.0304,0.0576,100)
        contrasts = np.random.normal(0.8928,0.0805,100)
        saturations = np.random.normal(0.8441,0.1087,100)
        hues = np.random.normal(0.0014,0.0091,100)
        brightnesses = [m for m in brightnesses if 1.0304-2*0.0576 < m < 1.0304+2*0.0576]
        contrasts = [m for m in contrasts if 0.8928-2*0.0805 < m < 0.8928+2*0.0805]
        saturations = [m for m in saturations if 0.8441-2*0.1087 < m < 0.8441+2*0.1087]
        hues = [m for m in hues if 0.0014-2*0.0091 < m < 0.0014+2*0.0091]
        brightness = random.choice(brightnesses)
        contrast = random.choice(contrasts)
        saturation = random.choice(saturations)
        hue = random.choice(hues)

        img1s = tra.adjust_brightness(torch.from_numpy(img1s).permute(0, 3, 1, 2), brightness)#  0.9-1.1
        img1s = tra.adjust_saturation(img1s, contrast)#  1-1.2
        img1s = tra.adjust_contrast(img1s, saturation)#  1-1.2
        img1s = tra.adjust_hue(img1s, hue).permute(0, 2, 3, 1).numpy()#  -0.01-0.01
        gts = tra.adjust_brightness(torch.from_numpy(gts).permute(0, 3, 1, 2), brightness*t + 1 - t)#  0.9-1.1
        gts = tra.adjust_saturation(gts, contrast*t + 1 - t)#  1-1.2
        gts = tra.adjust_contrast(gts, saturation*t + 1 - t)#  1-1.2
        gts = tra.adjust_hue(gts, hue * t).permute(0, 2, 3, 1).numpy()#  -0.01-0.01
        
        if self.dataset_name == 'train':
            if random.uniform(0, 1) < 0.5:  
                img0s = img0s[:, :, :, ::-1]
                img1s = img1s[:, :, :, ::-1]
                gts = gts[:, :, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, ::-1]
                img1s = img1s[:, ::-1]
                gts = gts[:, ::-1]

            if random.uniform(0, 1) < 0.5:
                img0s = img0s[:, :, ::-1]
                img1s = img1s[:, :, ::-1]
                gts = gts[:, :, ::-1]
                
        img0s = torch.from_numpy(img0s.copy()).permute(0, 3, 1, 2).float()
        img1s = torch.from_numpy(img1s.copy()).permute(0, 3, 1, 2).float()
        gts = torch.from_numpy(gts.copy()).permute(0, 3, 1, 2)
        
        timesteps = torch.tensor(timesteps).reshape(gts.shape[0], 1, 1, 1)

        return torch.cat((img0s, img1s, gts), 1), timesteps