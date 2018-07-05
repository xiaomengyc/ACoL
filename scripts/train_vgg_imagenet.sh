#!/bin/sh

cd ../exper/


ROOT_DIR=/home/zhangxiaolin/xlzhang/acol
IMG_DIR=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
#IMG_DIR=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
TRAIN_LIST=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
#TRAIN_LIST=${ROOT_DIR}/data/ILSVRC/list/val_list.txt \
THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=0,1 python train_frame.py --arch=vgg_v2 --epoch=6 --lr=0.0001 --batch_size=30 --num_gpu=2 --dataset=imagenet  \
	--img_dir=${IMG_DIR} \
	--disp_interval=40 \
	--train_list=${TRAIN_LIST} \
	--num_classes=1000 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--snapshot_dir=../snapshots/vgg_imagenet_v2_caffe_stg3/  \
	--restore_from=/home/zhangxiaolin/.torch/models/deeplab_vgg16_20M.pth \
#	--restore_from=../snapshots/vgg_imagenet_v2_caffe/imagenet_epoch_0_glo_step_42706.pth.tar \
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
#	--restore_from=/home/xiaolin/.torch/models/vgg16-00b39a1b-caffe.pth \
	#--restore_from=/home/xiaolin/.torch/models/inception_v3_google-1a9a5a14.pth \
