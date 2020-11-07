import torch
import torch.nn as nn
from torchvision.ops import RoIPool
from models.utils import *


class VGG16RoIHead(nn.Module):
    '''
    基于VGG16实现的Faster RCNN Head
    Args:
        n_class(int): 包括背景类在内的类别总数
        roi_size(int): feature map经过ROIPooling后的宽高
        spatial_scale(foalt): roi被缩放的倍数
        classifier(nn.Moudle): 来自VGG16的两个线性层
    '''
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        # 网络结构
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        # 初始化网络参数
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        # 其他参数
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        # RoI池化层
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        '''
        RoIHead 前向传播
        :param x: (Variable): (N, C , H, W)
        :param rois: (tensor) (R, 4) 候选区坐标数组
        :param roi_indices: 包含ROI对应的图像索引的数组
        :return:
        '''
        # 当roi_indices 是ndarray
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)   # [imgID, roi]
        # yx -> xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)    #([num_rois, C, 7, 7])
        pool = pool.view(pool.size(0), -1)      #([num_rois, C*7*7])
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        print("ROIHead return :")
        print("roi_cls_locs.size:", roi_cls_locs.size())
        print("roi_scores.size:", roi_scores.size())
        return roi_cls_locs, roi_scores