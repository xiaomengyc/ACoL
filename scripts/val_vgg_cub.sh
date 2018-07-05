#!/bin/sh

cd ../exper/


ROOT_DIR=/home/xiaolin/xlzhang/ACoL
IMG_DIR=${ROOT_DIR}/data/CUB_200_2011/images \
TEST_LIST=${ROOT_DIR}/data/CUB_200_2011/list/test.txt \
THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=1 python val_frame.py --arch=vgg_v1 --epoch=31 --lr=0.0001 --batch_size=20 --num_gpu=1 --dataset=imagenet  \
	--img_dir=${IMG_DIR} \
	--test_list=${TEST_LIST} \
	--num_classes=200 \
	--threshold=${THRESHOLD} \
	--num_workers=6 \
	--snapshot_dir=../snapshots/vgg_cub_v1/  \
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
	#--restore_from=/home/xiaolin/.torch/models/inception_v3_google-1a9a5a14.pth \
