#!/bin/sh

ROOT_DIR=`pwd`/..

cd ../exper/


#ROOT_DIR=/home/zhangxiaolin/xlzhang/acol
IMG_DIR=${ROOT_DIR}/data/COCO14/images
TRAIN_LIST=${ROOT_DIR}/data/COCO14/list/train_onehot.txt 
TEST_LIST=${ROOT_DIR}/data/COCO14/list/val_onehot.txt 
#TRAIN_LIST=${ROOT_DIR}/data/ILSVRC/list/val_list.txt \
THRESHOLD=0.8


CUDA_VISIBLE_DEVICES=0,1,3 python val_frame.py --arch=vgg_v0  --batch_size=60 --num_gpu=1 --dataset=imagenet  \
	--img_dir=${IMG_DIR} \
	--disp_interval=40 \
	--test_list=${TRAIN_LIST} \
	--num_classes=80 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--num_gpu=3 \
	--snapshot_dir=../snapshots/vgg_coco_v0/  \
#--restore_from=${HOME}/.torch/models/vgg16-397923af.pth \
#--restore_from=/home/zhangxiaolin/.torch/models/deeplab_vgg16_20M.pth \
#	--restore_from=../snapshots/vgg_imagenet_v2_caffe/imagenet_epoch_0_glo_step_42706.pth.tar \
#	--restore_from=/home/xiaolin/.torch/models/vgg16-00b39a1b-caffe.pth \
	#--restore_from=/home/xiaolin/.torch/models/inception_v3_google-1a9a5a14.pth \
