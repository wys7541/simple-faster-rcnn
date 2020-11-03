import torch
import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

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