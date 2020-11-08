import torch
import numpy as np

def smooth_l1_loss(x, t, in_weight, sigma):
    '''求smooth_l1_loss'''
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    '''求loc loss'''
    in_weight = torch.zeros(gt_loc.shape)
    # 仅对postive ROI计算局部化损失
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 通过正负ROI的总数进行标准化, gt_label=-1对rpn loss没有影响
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss


