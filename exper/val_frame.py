import sys
sys.path.append('../')

import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import os
from tqdm import tqdm
import time
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader
import shutil
import collections
from torch.optim import SGD
import my_optim
import torch.nn.functional as F
from models import *
from torch.autograd import Variable
import torchvision
from utils import AverageMeter
from utils import Metrics
from utils.save_atten import SAVE_ATTEN
from utils.LoadData import data_loader2, data_loader
from utils.Restore import restore

ROOT_DIR = '/'.join(os.getcwd().split('/')[:-1])
print 'Project Root Dir:',ROOT_DIR

IMG_DIR=os.path.join(ROOT_DIR,'data','ILSVRC','Data','CLS-LOC','train')
SNAPSHOT_DIR=os.path.join(ROOT_DIR,'snapshot_bins')

train_list = os.path.join(ROOT_DIR,'datalist','train_list.txt')
test_list = os.path.join(ROOT_DIR,'datalist','val_list.txt')


LR = 0.001
# LR=0.1
EPOCH = 200
DISP_INTERVAL = 50
def get_arguments():
    parser = argparse.ArgumentParser(description='ACoL')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--train_list", type=str, default=train_list)
    parser.add_argument("--test_list", type=str, default=test_list)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--arch", type=str,default='vgg_v1')
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='True')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args, threshold=args.threshold)

    model = torch.nn.DataParallel(model, range(args.num_gpu))
    model.cuda()

    optimizer = my_optim.get_optimizer(args, model)

    if args.resume == 'True':
        restore(args, model, optimizer)


    return  model, optimizer

def val(args, model=None, current_epoch=0):
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1.reset()
    top5.reset()

    if model is None:
        model, _ = get_model(args)
    model.eval()
    train_loader, val_loader = data_loader(args, test_path=True)

    save_atten = SAVE_ATTEN(save_dir='../save_bins/')

    global_counter = 0
    prob = None
    gt = None
    for idx, dat  in tqdm(enumerate(val_loader)):
        img_path, img, label_in = dat
        global_counter += 1
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)
            label_input = label_in.repeat(10, 1)
            label = label_input.view(-1)
        else:
            label = label_in

        img, label = img.cuda(), label.cuda()
        img_var, label_var = Variable(img), Variable(label)

        logits = model(img_var, label_var)

        logits0 = logits[0]
        if args.tencrop == 'True':
            logits0 = logits0.view(bs, ncrops, -1).mean(1)


        prec1_1, prec5_1 = Metrics.accuracy(logits0.cpu().data, label_in.long(), topk=(1,5))
        # prec3_1, prec5_1 = Metrics.accuracy(logits[1].data, label.long(), topk=(1,5))
        top1.update(prec1_1[0], img.size()[0])
        top5.update(prec5_1[0], img.size()[0])

        # model.module.save_erased_img(img_path)
        last_featmaps = model.module.get_localization_maps()
        np_last_featmaps = last_featmaps.cpu().data.numpy()

        # Save 100 sample masked images by heatmaps
        if idx < 100/args.batch_size:
            save_atten.get_masked_img(img_path, np_last_featmaps, label_in.numpy(), size=(0,0), maps_in_dir=False)

        # save_atten.get_masked_img(img_path, np_last_featmaps, label_in.numpy(),size=(0,0),
        #                           maps_in_dir=True, save_dir='../heatmaps',only_map=True )

        # np_scores, pred_labels = torch.topk(logits0,k=args.num_classes,dim=1)
        # pred_np_labels = pred_labels.cpu().data.numpy()
        # save_atten.save_top_5_pred_labels(pred_np_labels[:,:5], img_path, global_counter)
        # # pred_np_labels[:,0] = label.cpu().numpy() #replace the first label with gt label
        # # save_atten.save_top_5_atten_maps(np_last_featmaps, pred_np_labels, img_path)



    if args.onehot=='True':
        print val_mAP
        print 'AVG:', np.mean(val_mAP)

    else:
        print('Top1:', top1.avg, 'Top5:',top5.avg)


    # save_name = os.path.join(args.snapshot_dir, 'val_result.txt')
    # with open(save_name, 'a') as f:
    #     f.write('%.3f'%out)


if __name__ == '__main__':
    args = get_arguments()
    import json
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
