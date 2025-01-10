""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
import torch.nn.functional as F

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def pairwise_dist(A, B):
    na = torch.sum(A**2, dim=1)
    nb = torch.sum(B**2, dim=1)

    na = na.reshape(-1, 1)
    nb = nb.reshape(1, -1)

    D = torch.sqrt(torch.maximum(na - 2 * torch.matmul(A, B.T) + nb, torch.tensor(1e-12)))
    return D

def cross_entropy_nn_pred(enc_x, in_y, learnt_y):
    """Cross Entropy NN Prediction based on learnt_y."""

    enc_x_to_learnt_y_dist = pairwise_dist(enc_x, learnt_y)
    logits = F.softmax(-1. * enc_x_to_learnt_y_dist, dim=1)
    # print('logits', logits.shape)
    preds = torch.argmax(logits, dim=1)
    # print('in_y', in_y.shape)
    # print('in_y values', in_y)

    true_y = torch.argmax(in_y, dim=1)
    return preds, true_y

def lwal_accuracy(output, target, learnt_y, topk=(1,)):
    """Computes the 1-accuracy for lwal loss."""
    # print('output', output.type())
    x = output.to(torch.float32)
    # print('x', x.type())
    # print('target', target.type())
    # print('learnt_y', learnt_y.type())
    one_hot_target = torch.nn.functional.one_hot(target, num_classes=10)
    pred_y, true_y = cross_entropy_nn_pred(x, one_hot_target, learnt_y)
    acc1 = (pred_y == true_y).float().mean() * 100.
    return acc1, 0.0