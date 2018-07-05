#!/bin/sh

cd ../exper/


ROOT_DIR=/home/zhangxiaolin/xlzhang/acol
IMG_DIR=${ROOT_DIR}/data/IMAGENET_VOC_3W/imagenet_simple \
TRAIN_LIST=${ROOT_DIR}/data/IMAGENET_VOC_3W/list/train.txt \
THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=0,1 python train_frame.py --arch=vgg_v2 --epoch=11 --lr=0.0001 --batch_size=30 --num_gpu=2 --dataset=imagenet  \
	--img_dir=${IMG_DIR} \
	--disp_interval=20 \
	--train_list=${TRAIN_LIST} \
	--num_classes=1000 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--snapshot_dir=../snapshots/vgg_3w_v2_caffe/  \
	--restore_from=/home/zhangxiaolin/.torch/models/deeplab_vgg16_20M.pth \
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
#	--restore_from=/home/xiaolin/.torch/models/vgg16-00b39a1b-caffe.pth \
	#--restore_from=/home/xiaolin/.torch/models/inception_v3_google-1a9a5a14.pth \
