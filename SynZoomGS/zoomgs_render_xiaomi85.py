#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import time
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import utils.Spline as Spline
import torch.nn as nn
import random
import math
# import cv2

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

class WideCamera(nn.Module):
    def __init__(self, camrea):
        super(WideCamera, self).__init__()

        self.uid = camrea.uid
        self.colmap_id = camrea.colmap_id
        self.R = camrea.R
        self.T = camrea.T
        self.FoVx = camrea.FoVx
        self.FoVy = camrea.FoVy

        self.image_name = camrea.image_name
        self.image = camrea.image

        try:
            self.data_device = torch.device(camrea.data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {camrea.data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = camrea.original_image
        self.image_width = 1216#camrea.image_width
        self.image_height = 1632#camrea.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = camrea.trans
        self.scale = camrea.scale

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update(self, ):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def generate_linear_camera(start_camera, end_camera, N_C, N_I):
    linear_cameras = []

    start_pose = np.concatenate([np.array(start_camera.R), np.expand_dims(np.array(start_camera.T),-1)], axis=-1) 
    start_pose = torch.Tensor(start_pose).unsqueeze(dim=0)

    end_pose = np.concatenate([np.array(end_camera.R), np.expand_dims(np.array(end_camera.T),-1)], axis=-1) 
    end_pose = torch.Tensor(end_pose).unsqueeze(dim=0)

    poses_start_se3 = Spline.SE3_to_se3_N(start_pose[:, :3, :4])  
    poses_end_se3 = Spline.SE3_to_se3_N(end_pose[:, :3, :4])     

    pose_nums = torch.cat([torch.zeros((1, N_C)), torch.arange(N_I).reshape(1, -1)], -1).repeat(poses_start_se3.shape[0], 1) 

    seg_pos_x = torch.arange(poses_start_se3.shape[0]).reshape([poses_start_se3.shape[0], 1]).repeat(1, N_C+N_I)
    se3_start = poses_start_se3[seg_pos_x, :]   
    se3_end = poses_end_se3[seg_pos_x, :]       

    spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, N_I).cpu().numpy()

    nums = N_C + N_I
    pose_nums = torch.arange(N_C + N_I).reshape(1, -1)
    pose_time = pose_nums[0] / (nums - 1)
    start_FovX = np.array(start_camera.FoVx)
    start_FovY = np.array(start_camera.FoVy)

    end_FovX = np.array(end_camera.FoVx)
    end_FovY = np.array(end_camera.FoVy)

    l_FovX =  (1 - pose_time) * start_FovX + pose_time * end_FovX    
    l_FovY =  (1 - pose_time) * start_FovY + pose_time * end_FovY 

    l_FovX = l_FovX.cpu().numpy()
    l_FovY = l_FovY.cpu().numpy()

    linear_cameras.append(start_camera)
    for kk in range(1, nums-1):
        cam = WideCamera(Camera(colmap_id=start_camera.colmap_id, R=spline_poses[kk, :3, :3], T=spline_poses[kk, :3, 3], 
                            FoVx=l_FovX[kk], FoVy=l_FovY[kk], gt_alpha_mask=None, image=start_camera.image,
                            image_name=start_camera.image_name, uid=start_camera.uid,
                            trans=start_camera.trans, scale=start_camera.scale, data_device=start_camera.data_device)
        )
        linear_cameras.append(cam)
    return linear_cameras

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    #ids = [1,50,100,150,200,250,280]
    ids = [1,100,200,300]
    print('#####################################')
    print("rendering xiaomi 85")
    print('#####################################')
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx in ids:
            if view.T[2] > 0.6 or view.T[2] < -0.6:
                N_C = 0
                N_I = 33  
                # To release FI burden, x0.6 ~ 0.85 can be implemented by SR, 0.85 ~ 1.0 can beimplemented by continuous camera transition 
                # Thus N_C camera encodings are set to 0., and N_I camera encodings are set to (0., 1.)
                c_views = np.concatenate((np.zeros(N_C), np.linspace(0., 1., N_I)), 0)
                
                # xiaomi:
                xs = np.random.normal(-0.2857,0.1369,100)
                ys = np.random.normal(-0.3828,0.2596,100)
                zs = np.random.normal(0.1931,0.1550,100)
                xs = [m for m in xs if -0.2857-2*0.1369 < m < -0.2857+2*0.1369]
                ys = [m for m in ys if -0.3828-2*0.2596 < m < -0.3828+2*0.2596]
                zs = [m for m in zs if 0.1931-2*0.1550 < m < 0.1931+2*0.1550]
                x = random.choice(xs) #* 0.4
                y = random.choice(ys) #* 0.4
                z = random.choice(zs) #* 0.4
                z = -z

                print('###########################')
                print(view.T )
                print(view.FoVx, view.FoVy)
                print(x, y, z)
                print('###########################')

                fovx_uw = math.tan(view.FoVx * 0.5)
                fovy_uw = math.tan(view.FoVy * 0.5)
                fovx_uw = fovx_uw * (1216 / view.image_width) * 0.7
                fovy_uw = fovy_uw * (1632 / view.image_height) * 0.7
                view.FoVx = math.atan(fovx_uw) * 2
                view.FoVy = math.atan(fovy_uw) * 2

                fovx_w = 0.9*math.tan(view.FoVx * 0.5)
                fovy_w = 0.9*math.tan(view.FoVy * 0.5)

                w_FoVx = math.atan(fovx_w) * 2
                w_FoVy = math.atan(fovy_w) * 2

                view.image_width = 1216
                view.image_height = 1632

                w_view = Camera(colmap_id=view.colmap_id, R=view.R, T=view.T + [x*(0.15/0.4),y*(0.15/0.4),z*(0.15/0.4)] , FoVx=w_FoVx, FoVy=w_FoVy , image=view.image, gt_alpha_mask=None, 
                                image_name=view.image_name, uid=view.uid)
                
                linear_views = generate_linear_camera(WideCamera(view), WideCamera(w_view), N_C, N_I)

                makedirs(os.path.join(render_path, str(idx)+'_x85'), exist_ok=True)
                linear_views = linear_views[N_C:]
                c_views = c_views[N_C:]
                for ii in range(0, len(linear_views)):
                    view0 = linear_views[ii]
                    c = c_views[ii]
                    
                    rendering = render(view0, gaussians, pipeline, background)["render"]
                    #render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
                    torchvision.utils.save_image(rendering, os.path.join(render_path, str(idx)+'_x85',  '%04d.png'%(ii+1)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)