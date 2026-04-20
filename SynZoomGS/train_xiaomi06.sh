#!/bin/bash

echo "Start to train the model...."

dataroot="/hdd3/fanjunjie/dl3dv_dataset_chuban/dl3dv_dataset/"
output="/home/ubuntu/Desktop/HoGS-main/output_1knew1/"
enddir="/gaussian_splat"

if [ ! -d "output" ]; then
        mkdir $output
fi

softfiles=$(ls $dataroot)
#softfiles=('01 02 03') #  scene id

echo $softfiles
for sfile in $softfiles
do 
    scene=$dataroot$sfile
    outputdir=$output$sfile
    echo $scene
    echo $outputdir
    # # #  pretrain base gs with uw image
    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train.py  \
     -s $scene -m $outputdir --port 6014 -r 1
    # # generate camera transition sequences
    CUDA_VISIBLE_DEVICES=0 python zoomgs_render_xiaomi06.py -m $outputdir -s $scene
    
done
