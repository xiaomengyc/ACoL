#!/bin/sh

cd ../exper/


ROOT_DIR=/home/xiaolin/acol
IMG_DIR=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
TRAIN_LIST=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
TEST_LIST=${ROOT_DIR}/data/ILSVRC/list/val_list.txt \
THRESHOLD=0.6


CUDA_VISIBLE_DEVICES=0 python val_frame.py --arch=vgg_v1   --batch_size=1 --num_gpu=1 --dataset=imagenet  \
	--img_dir=${IMG_DIR} \
	--test_list=${TEST_LIST} \
	--num_classes=1000 \
	--threshold=${THRESHOLD} \
	--tencrop=False \
	--num_workers=6 \
	--snapshot_dir=../snapshots/vgg_imagenet_v1/  \
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \
	#--restore_from=/home/xiaolin/.torch/models/inception_v3_google-1a9a5a14.pth \
