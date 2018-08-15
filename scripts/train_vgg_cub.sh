#!/bin/sh

cd ../exper/


THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=0,1 python train_frame.py --arch=vgg_v1 --epoch=60 --lr=0.0001 --batch_size=20 --num_gpu=2 --dataset=cub  \
	--disp_interval=40 \
	--num_classes=200 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--train_list=../datalist/CUB/train_list.txt \
	--img_dir=../data/CUB_200_2011/images \
	--snapshot_dir=../snapshots/vgg_imagenet_v1/  \
	--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
