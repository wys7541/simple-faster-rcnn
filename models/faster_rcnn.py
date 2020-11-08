import torch
import torch.nn as nn
from config import opt
from models.utils import *

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        # mean std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def forward(self, x, scale=[1., 1.]):
        '''
        FasterRCNN前向传播
        :param x: (tensor)img
        :param scale: (list float) 2 number
        :return:
        '''
        img_size = x.shape[2:]
        print("img_size:", img_size)
        # 特征提取
        h = self.extractor(x)
        # RPN
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        # ROIHead
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    def get_optimizer(self):
        '''
        :return: 返回一个优化器，或者重写一个优化器
        '''
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def predict(self, imgs, sizes=None):
        '''
        从图片中检测目标
        :param imgs: (numpy.ndarray) CHW和RGB形式
        :param sizes:
        :return:
            Args：
                bboxes(list of float arrays)：(R, 4) (ymin, xmin, ymax, xmax)
                labels(list of integer array): (R, ) 类别
                scores(list of float array) (R, ) 置信度
        '''
        self.eval()
        bboxes = list()
        labels = list()
        scores = list()

        return bboxes, labels, scores
