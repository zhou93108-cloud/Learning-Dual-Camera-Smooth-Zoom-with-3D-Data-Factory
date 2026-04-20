# Learning-Dual-Camera-Smooth-Zoom-with-3D-Data-Factory
Code of Learning Dual-Camera Smooth Zoom with 3D Data Factory

## 1.Abstract
<img width="3084" height="815" alt="intro1" src="https://github.com/user-attachments/assets/ed361cf5-ea27-42d5-a0bc-ac09957990a9" />
When zooming between dual cameras on a mobile, noticeable jumps in geometric content and image color occur in the preview, inevitably affecting the user’s zoom experience. To address this, we introduce the dual-camera smooth zoom (DCSZ) task, aiming to synthesize intermediate frames for fluid zooming. However, naively applying existing frame interpolation (FI) models for the task is challenging due to the motion domain gap and the scarcity of real-world ground truth. In this paper, we propose a novel data factory based on 3D Gaussian Splatting (3DGS) to construct large-scale training data. Specifically, we introduce Syn-ZoomGS, which generates extensive data from camera-independent 3D models by sampling camera parameters from fitted distributions, and Real-ZoomGS, which achieves high-fidelity synthesis by decoupling scene geometry from camera-specific characteristics. Furthermore, we design ZoomFI, an effective FI network tailored for DCSZ that incorporates bidirectional optical flows for photo-realistic zooming. Extensive experiments on both synthetic and real-world datasets of two mobile phones demonstrate that fine-tuning with our constructed DCSZ data significantly improves the performance of FI methods. Moreover, the proposed ZoomFI achieves state-of-the-art results in both quantitative metrics and visual quality. The datasets, codes, and pre-trained models will be publicly available.

## 2.Method
### 2.1 Syn-ZoomGS
<img width="3284" height="1011" alt="pipline_2" src="https://github.com/user-attachments/assets/2996e5bd-b140-426e-b8b0-c82c15e59820" />
First we use the Syn-ZoomGS method to generate training data. (a)The pipline of Syn-ZoomGS. Syn-ZoomGS first sample camera parameters of UW and W from data Distribution, calculate and interpolate camera parameter. Then it render image sequence from reconstructed 3DGS representation. Finally it samples color transformation parameters from color Distribution and interpolates them, and applies them to the rendered images. (b) Statistics of Geometric transformation parameters. (c) Statistics of Color transformation parameters.

### 2.2 Real-ZoomGS
<img width="6168" height="897" alt="pipline_3" src="https://github.com/user-attachments/assets/26a2cb15-9b74-449f-9248-4d7f65164e07" />
(a) Over view of Real-ZoomGS. The virtual (V) camera parameters are constructed by interpolating the dual-camera ones, and are then input into ZoomGS to generate zoom sequences. (b) Construction of Real-ZoomGS. Real-ZoomGS employs a camera transition (CamTrans) module to transform the base (\ie, UW camera) Gaussians to the specific camera Gaussians according to the camera encoding.

### 2.3 Zoom FI
<img width="357" height="433" alt="FI_model_0313" src="https://github.com/user-attachments/assets/f3e8a9fe-def5-42e9-ba77-acf6e2852277" />
The Structure of ZoomFI, a specialized FI model for photo-realistic zooming between dual cameras.

## 3.Prerequisites and Datasets
### 3.1 Prerequisites
- Python 3.8.16, PyTorch 2.1.1, **cuda-11.8**
- We provide detailed dependencies in [`environment.yml`] for Real-ZoomGS and ZoomFI, and [`./SynZoomGS/environment.yml`] for Syn-ZoomGS.

### 3.2 Datasets

### 3.3 Pretrained models

## 4.Start for Syn-ZoomGS
- Run [`cd ./SynZoomGS`](./SynZoomGS)
- For xiaomi data, zoom factor of W set to 0.6, 
  Run [`bash ./zoomgs_render_xiaomi06.sh`](./zoomgs_trains.sh)
- For xiaomi data, zoom factor of W set to 0.85, 
  Run [`bash ./zoomgs_render_xiaomi85.sh`](./zoomgs_trains.sh)
- For huawei data, zoom factor of W set to 0.6, 
  Run [`bash ./zoomgs_render_huawei06.sh`](./zoomgs_trains.sh)
- For huawei data, zoom factor of W set to 0.85, 
  Run [`bash ./zoomgs_render_huawei85.sh`](./zoomgs_trains.sh)

## 5.Start for Real-ZoomGS
- Run [`cd ./RealZoomGS`](./ZoomGS)
- Run [`bash ./zoomgs_train.sh`](./zoomgs_trains.sh)

## 6.Start for ZoomFI
- Run [`cd ./ZoomFI`](./FrameInterpolation)
- Training: run [`bash ./train.sh`](./train.sh)
- Testing on synthetic data: run [`bash ./test_syn.sh`](./test_syn.sh)
- Testing on real-world data: run [`bash ./test_real.sh`](./test_real.sh)

## Acknowledgement

Special thanks to the following awesome projects!

- [DCSZ](https://github.com/ZcsrenlongZ/ZoomGS)
- [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
- [DL3DV-10K](https://github.com/DL3DV-10K/Dataset)
- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [BAD-Gaussian](https://github.com/WU-CVGL/BAD-Gaussians)
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE)

## Citation
If you make use of our work, please cite our paper.
