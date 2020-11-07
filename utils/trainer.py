import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data.dataset import MyDataset
from config import opt
from models import FasterRCNNVGG16, AnchorTargetCreator, ProposalTargetCreator
from models.utils import *
from models.loss import *


class FasterRCNNTrainer(nn.Module):
    '''
    封装类方便训练，返回losses
    '''
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        # creator生成gt_bbox和gt_label作为训练目标
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

    def forward(self, imgs, bboxes, labels, scale):
        '''
        Faster RCNN的前向传播和计算损失
        :param imgs:
        :param bboxes:
        :param labels:
        :param scale:
        :return:
            Args：
                imgs(~torch.autograd.Variable)：批量的图片
                bboxes(~torch.autograd.Variable): (N, R, 4) 批量的bbox
                labels(~torch.autograd..Variable): (N, R) 批量的label
                scale(list of float): 预处理期间应用于原始图像的缩放量
        '''
        n = bboxes.shape[0]
        if n != 1:
            raise  ValueError('Currently only batch size 1 is supported.')
        _, _, H, W = imgs.shape
        img_size = (H, W)
        # 特征提取和RPN
        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
                self.faster_rcnn.rpn(features, img_size, scale)

        # 因为batch_size为1， 将变量转换成单数形式
        bbox = bboxes[0]
        label = labels[0]
        rpn_scores = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # sampe RoIs
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # head, index对应的图像索引的数组,因为batch_size=1所以全为0
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ----- RPN losses-----------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = totensor(gt_rpn_label).long()
        gt_rpn_loc = totensor(gt_roi_loc)
        rpn_loc_loss = fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)




    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)

        exit(0)




if __name__ == '__main__':
    train_data = MyDataset(opt)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers = 2)
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn)
    print('model construct completed')
    for epoch in range(opt.epoch):
        for ii, (img, bbox, label, scale) in enumerate(train_loader):
            scale = [scale[0].numpy()[0], scale[1].numpy()[0]]
            trainer.train_step(img, bbox, label, scale)

            exit(0)