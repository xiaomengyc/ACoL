#!/bin/sh

cd ../exper/


THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=0 python train_frame.py --arch=vgg_v1 --epoch=60 --lr=0.0001 --batch_size=20 --num_gpu=1 --dataset=cub  \
	--disp_interval=40 \
	--num_classes=1000 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--img_dir=../data/CUB_200_2011/images/ \
	--train_list=../datalist/CUB/train_list.txt \
	--snapshot_dir=../snapshots/vgg_imagenet_v1/  \
	--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
