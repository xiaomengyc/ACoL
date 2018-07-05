import numpy as np
import cv2
import os
import torch
import os
import time
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.autograd import Variable

idx2catename = {'voc20': ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse',
              'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'],

                'coco80': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                           'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                           'hair drier', 'toothbrush']}

class SAVE_ATTEN(object):
    def __init__(self, save_dir='save_bins', dataset=None):
        # type: (object, object) -> object
        self.save_dir = save_dir
        if dataset is not None:
            self.idx2cate = self._get_idx2cate_dict(datasetname=dataset)
        else:
            self.idx2cate = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_top_5_pred_labels(self, preds, org_paths, global_step):
        img_num = np.shape(preds)[0]
        for idx in xrange(img_num):
            img_name = org_paths[idx].strip().split('/')[-1]
            if '.JPEG' in img_name:
                img_id = img_name[:-5]
            elif '.png' in img_name or '.jpg' in img_name:
                img_id = img_name[:-4]

            out = img_id + ' ' + ' '.join(map(str, preds[idx,:])) + '\n'
            out_file = os.path.join(self.save_dir, 'pred_labels.txt')

            if global_step == 0 and idx==0 and os.path.exists(out_file):
                os.remove(out_file)
            with open(out_file, 'a') as f:
                f.write(out)

    def save_masked_img_batch(self, path_batch, atten_batch, label_batch):

        #img_num = np.shape(atten_batch)[0]
        img_num = atten_batch.size()[0]
        # fid = open('imagenet_val_shape.txt', 'a')
        # print(np.shape(img_batch), np.shape(label_batch), np.shape(org_size_batch), np.shape(atten_batch))
        for idx in xrange(img_num):
            atten = atten_batch[idx]
            atten = atten.cpu().data.numpy()
            label = label_batch[idx]
            label = int(label) 
            self._save_masked_img(path_batch[idx], atten,label)

    def _get_idx2cate_dict(self, datasetname=None):
        if datasetname not in idx2catename.keys():
            print 'The given %s dataset category names are not available. The supported are: %s'\
                  %(str(datasetname),','.join(idx2catename.keys()))
            return None
        else:
            return {idx:cate_name for idx, cate_name in enumerate(idx2catename[datasetname])}

    def _save_masked_img(self, img_path, atten, label):
        '''
        save masked images with only one ground truth label
        :param path:
        :param img:
        :param atten:
        :param org_size:
        :param label:
        :param scores:
        :param step:
        :param args:
        :return:
        '''
        if not os.path.isfile(img_path):
            raise 'Image not exist:%s'%(img_path)
        img = cv2.imread(img_path)
        org_size = np.shape(img)
        w = org_size[0]
        h = org_size[1]

        attention_map = atten[label,:,:]
        atten_norm = attention_map
        print(np.shape(attention_map), 'Max:', np.max(attention_map), 'Min:',np.min(attention_map))
       # min_val = np.min(attention_map)
       # max_val = np.max(attention_map)
       # atten_norm = (attention_map - min_val)/(max_val - min_val)
        atten_norm = cv2.resize(atten_norm, dsize=(h,w))
        atten_norm = atten_norm* 255
        heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

        img_id = img_path.strip().split('/')[-1]
        img_id = img_id.strip().split('.')[0]
        save_dir = os.path.join(self.save_dir, img_id+'.png')
        cv2.imwrite(save_dir, img)


    def save_top_5_atten_maps(self, atten_fuse_batch, top_indices_batch, org_paths, topk=5):
        '''
        Save top-5 localization maps for generating bboxes
        :param atten_fuse_batch: normalized last layer feature maps of size (batch_size, C, W, H), type: numpy array
        :param top_indices_batch: ranked predicted labels of size (batch_size, C), type: numpy array
        :param org_paths:
        :param args:
        :return:
        '''
        img_num = np.shape(atten_fuse_batch)[0]
        for idx in xrange(img_num):
            img_id = org_paths[idx].strip().split('/')[-1][:-4]
            for k in range(topk):
                atten_pos = top_indices_batch[idx, k]
                atten_map = atten_fuse_batch[idx, atten_pos,:,:]
                heat_map = cv2.resize(atten_map, dsize=(224, 224))
                # heat_map = cv2.resize(atten_map, dsize=(img_shape[1], img_shape[0]))
                heat_map = heat_map* 255
                save_path = os.path.join(self.save_dir, 'heat_maps', 'top%d'%(k+1))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path,img_id+'.png')
                cv2.imwrite(save_path, heat_map)

    def normalize_map(self, atten_map):
        min_val = np.min(atten_map)
        max_val = np.max(atten_map)
        atten_norm = (atten_map - min_val)/(max_val - min_val)

        return atten_norm

    def _add_msk2img(self, img, msk, isnorm=True):
        if np.ndim(img) == 3:
            assert np.shape(img)[0:2] == np.shape(msk)
        else:
            assert np.shape(img) == np.shape(msk)

        if isnorm:
            min_val = np.min(msk)
            max_val = np.max(msk)
            atten_norm = (msk - min_val)/(max_val - min_val)
        atten_norm = atten_norm* 255
        heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
        w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)

        return w_img

    def _draw_text(self, pic, txt, pos='topleft'):
        font = cv2.FONT_HERSHEY_SIMPLEX   #multiple line
        txt = txt.strip().split('\n')
        stat_y = 30
        for t in txt:
            pic = cv2.putText(pic,t,(10,stat_y), font, 0.8,(255,255,255),2,cv2.LINE_AA)
            stat_y += 30

        return pic


    def _mark_score_on_picture(self, pic, score_vec, label_idx):
        score = score_vec[label_idx]
        txt = '%.3f'%(score)
        pic = self._draw_text(pic, txt, pos='topleft')
        return pic


    def get_heatmap_idxes(self, gt_label):

        labels_idx = []
        if np.ndim(gt_label) == 1:
            labels_idx = np.expand_dims(gt_label, axis=1).astype(np.int)
        elif np.ndim(gt_label) == 2:
            for row in gt_label:
                idxes = np.where(row[0]==1)[0] if np.ndim(row)==2 else np.where(row==1)[0]
                labels_idx.append(idxes.tolist())
        else:
            labels_idx = None

        return labels_idx

    def get_map_k(self, atten, k, size=(224,224)):
        atten_map_k = atten[k,:,:]
        # print np.max(atten_map_k), np.min(atten_map_k)
        atten_map_k = cv2.resize(atten_map_k, dsize=size)
        return atten_map_k

    def read_img(self, img_path, size=(224,224)):
        img = cv2.imread(img_path)
        if img is None:
            print "Image does not exist. %s" %(img_path)
            exit(0)

        if size == (0,0):
            size = np.shape(img)[:2]
        else:
            img = cv2.resize(img, size)
        return img, size[::0]


    def get_masked_img(self, img_path, atten, gt_label,
                       size=(224,224), score_vec=None, maps_in_dir=False, save_dir=None, only_map=False):

        assert np.ndim(atten) == 4

        save_dir = save_dir if save_dir is not None else self.save_dir

        if isinstance(img_path, list) or isinstance(img_path, tuple):
            batch_size = len(img_path)
            label_indexes = self.get_heatmap_idxes(gt_label)
            for i in range(batch_size):
                img, size = self.read_img(img_path[i], size)
                img_name = img_path[i].split('/')[-1]
                img_name = img_name.strip().split('.')[0]
                if maps_in_dir:
                    img_save_dir = os.path.join(save_dir, img_name)
                    os.mkdir(img_save_dir)

                for k in label_indexes[i]:
                    atten_map_k = self.get_map_k(atten[i], k , size)
                    msked_img = self._add_msk2img(img, atten_map_k)

                    suffix = str(k)
                    if only_map:
                        save_img = (self.normalize_map(atten_map_k)*255).astype(np.int)
                    else:
                        save_img = msked_img

                    if maps_in_dir:
                        cv2.imwrite(os.path.join(img_save_dir, suffix + '.png'), save_img)
                    else:
                        cv2.imwrite(os.path.join(save_dir, img_name + '_' + suffix + '.png'), save_img)

                    #     if score_vec is not None and labels_idx is not None:
                    #         msked_img = self._mark_score_on_picture(msked_img, score_vec, labels_idx[k])
                    #     if labels_idx is not None:
                    #         suffix = self.idx2cate.get(labels_idx[k], k)


    # def get_masked_img_ml(self, img_path, atten, save_dir=None, size=(224,224),
    #                       gt_label=None, score_vec=None):
    #     assert np.ndim(atten) == 4
    #
    #     if gt_label is not None and self.idx2cate is not None:
    #         labels_idx = np.where(gt_label[0]==1)[0] if np.ndim(gt_label)==2 else np.where(gt_label==1)[0]
    #     else:
    #         labels_idx = None
    #
    #
    #     if save_dir is not None:
    #         self.save_dir = save_dir
    #     if isinstance(img_path, list) or isinstance(img_path, tuple):
    #         batch_size = len(img_path)
    #         for i in range(batch_size):
    #             img = cv2.imread(img_path[i])
    #             if img is None:
    #                 print "Image does not exist. %s" %(img_path[i])
    #                 exit(0)
    #
    #             else:
    #                 atten_img = atten[i]     #get attention maps for the i-th img
    #                 img_name = img_path[i].split('/')[-1]
    #                 for k in range(np.shape(atten_img)[0]):
    #                     if size == (0,0):
    #                         w, h, _ = np.shape(img)
    #                         # h, w, _ = np.shape(img)
    #                     else:
    #                         h, w = size
    #                         img = cv2.resize(img, dsize=(h, w))
    #                     atten_map_k = atten_img[k,:,:]
    #                     # print np.max(atten_map_k), np.min(atten_map_k)
    #                     atten_map_k = cv2.resize(atten_map_k, dsize=(h,w))
    #                     msked_img = self._add_msk2img(img, atten_map_k)
    #                     if score_vec is not None and labels_idx is not None:
    #                         msked_img = self._mark_score_on_picture(msked_img, score_vec, labels_idx[k])
    #                     if labels_idx is not None:
    #                         suffix = self.idx2cate.get(labels_idx[k], k)
    #                     else:
    #                         suffix = str(k)
    #                     if '.' in img_name:
    #                         img_name = img_name.strip().split('.')[0]
    #                     cv2.imwrite(os.path.join(self.save_dir, img_name + '_' + suffix + '.png'), msked_img)
    #
    #
    # def get_masked_img(self, img_path, atten, save_dir=None,  size=(224,224), combine=True):
    #     '''
    #
    #     :param img_path:
    #     :param atten:
    #     :param size: if it is (0,0) use original image size, otherwise use the specified size.
    #     :param combine:
    #     :return:
    #     '''
    #
    #     if save_dir is not None:
    #         self.save_dir = save_dir
    #     if isinstance(img_path, list) or isinstance(img_path, tuple):
    #         batch_size = len(img_path)
    #
    #         for i in range(batch_size):
    #             atten_norm = atten[i]
    #             min_val = np.min(atten_norm)
    #             max_val = np.max(atten_norm)
    #             atten_norm = (atten_norm - min_val)/(max_val - min_val)
    #             # print np.max(atten_norm), np.min(atten_norm)
    #             img = cv2.imread(img_path[i])
    #             if img is None:
    #                 print "Image does not exist. %s" %(img_path[i])
    #                 exit(0)
    #
    #             if size == (0,0):
    #                 w, h, _ = np.shape(img)
    #                 # h, w, _ = np.shape(img)
    #             else:
    #                 h, w = size
    #                 img = cv2.resize(img, dsize=(h, w))
    #
    #             atten_norm = cv2.resize(atten_norm, dsize=(h,w))
    #             # atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    #             atten_norm = atten_norm* 255
    #             heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
    #             img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
    #
    #
    #             # font = cv2.FONT_HERSHEY_SIMPLEX
    #             # cv2.putText(img,'OpenCV \n hello',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
    #
    #             img_name = img_path[i].split('/')[-1]
    #             print os.path.join(self.save_dir, img_name)
    #             cv2.imwrite(os.path.join(self.save_dir, img_name), img)

    def get_atten_map(self, img_path, atten, save_dir=None,  size=(321,321)):
        '''
        :param img_path:
        :param atten:
        :param size: if it is (0,0) use original image size, otherwise use the specified size.
        :param combine:
        :return:
        '''

        if save_dir is not None:
            self.save_dir = save_dir
        if isinstance(img_path, list) or isinstance(img_path, tuple):
            batch_size = len(img_path)

            for i in range(batch_size):
                atten_norm = atten[i]
                min_val = np.min(atten_norm)
                max_val = np.max(atten_norm)
                atten_norm = (atten_norm - min_val)/(max_val - min_val)
                # print np.max(atten_norm), np.min(atten_norm)
                h, w = size

                atten_norm = cv2.resize(atten_norm, dsize=(h,w))
                # atten_norm = cv2.resize(atten_norm, dsize=(w,h))
                atten_norm = atten_norm* 255

                img_name = img_path[i].split('/')[-1]
                img_name = img_name.replace('jpg', 'png')
                cv2.imwrite(os.path.join(self.save_dir, img_name), atten_norm)

class DRAW(object):
    def __init__(self):
        pass

    def draw_text(self, img, text):
        if isinstance(text, dict):
            pass
