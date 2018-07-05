import torch
import cv2
import numpy as np

def accuracy(logits, target, topk=(1,)):
    '''
    Compute the top k accuracy of classification results.
    :param target: the ground truth label
    :param topk: tuple or list of the expected k values.
    :return: A list of the accuracy values. The list has the same lenght with para: topk
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    scores = logits

    _, pred = scores.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


from sklearn import metrics
def get_mAP(gt_labels, pred_scores):
    n_classes = np.shape(gt_labels)[1]
    results = []
    for i in range(n_classes):
        res = metrics.average_precision_score(gt_labels[:,i], pred_scores[:,i])
        results.append(res)

    results = map(lambda x: '%.3f'%(x), results)
    cls_map = np.array(map(float, results))
    return cls_map

def get_AUC(gt_labels, pred_scores):
    res = metrics.roc_auc_score(gt_labels, pred_scores)
    return res

def _to_numpy(v):
    v = torch.squeeze(v)
    if torch.is_tensor(v):
        v = v.cpu()
        v = v.numpy()
    elif isinstance(v, torch.autograd.Variable):
        v = v.cpu().data.numpy()

    return v

def get_iou(pred, gt):
    '''
    IoU which is averaged by images
    :param pred:
    :param gt:
    :return:
    '''
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    pred[gt==255] = 255

    assert pred.shape == gt.shape

    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    # max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((20 + 1,))
    for j in range(20 + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    unique_classes = len(np.unique(gt))-1 if 255 in np.unique(gt).tolist() else len(np.unique(gt))
    # unique_classes = len(np.unique(gt))
    Aiou = np.sum(result_class[:]) / float(unique_classes)

    return Aiou

def fast_hist(pred, gt, n=21):
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * pred[k].astype(int) + gt[k], minlength=n**2).reshape(n, n)

def get_voc_iou(hist):
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return miou

