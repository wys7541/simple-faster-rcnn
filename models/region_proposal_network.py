import torch
import torch.nn as nn
import torch.nn.functional as F
from config import opt
from models.utils import *

class RegionProposalNetwork(nn.Module):
    '''RPN
    Args:
        in_channels(int): 输入的通道数
        mid_channels(int): 中间张量的通道数
        ratios(list of float): anchor的宽高比例列表
        anchor_scales(list of number): 窗口的缩放倍数
        feat_stride(int): 下采样倍数
        proposal_creator_params(dict): 键值对参数，用于class:`model.utils.creator_tools.ProposalCreator`.
    '''
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1., 2.],
                 anchor_scales=[8, 16, 32], feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        # 生成图片右上角的第一个anchor_base
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_strde = feat_stride
        # 候选区生成
        self.proposal_layer = ProposalCreator()
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # intermediate layer
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # cls layer
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)   # reg layer
        # 参数初始化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale):
        '''
        RPN的前向传播
        :param x: (~torch.autograd.Variable) (N, C, H, W)
        :param img_size: (tuple of ints) (height, width)
        :param scale: (list of float) (height / ori_height, width / ori_width)
        :return:
        '''
        n, _, hh, ww = x.shape
        print(x.shape)
        # anchor 最初的（height*width*9, 4）个bbox
        anchor = enumerate_shifted_anchor(np.array(self.anchor_base),
                                          self.feat_strde, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        h = F.relu(self.conv1(x))   # intermediate layer

        # rpn_locs 表示对anchor的预测边界框偏移和比例
        rpn_locs = self.loc(h)      # cls layer
        # rpn_locs ([1, 36, 20, 30]) -> ([1, 5400, 4])
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(h)  # reg layer
        # dim=2表示label negtive和posttion的概率
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_pos_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_pos_scores = rpn_pos_scores.view(n, -1)     # ([1, 5400])
        rpn_scores = rpn_scores.view(n, -1, 2)          # ([1, 5400, 2])

        rois = list()
        rois_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),
                                      rpn_pos_scores[i].cpu().data.numpy(),
                                      anchor, img_size, scale)
            print(roi)
        exit(0)






class ProposalCreator:
    '''调用此类生成候选区
    Args：
        nms_thresh(float): 调用NMS时的阈值
        n_train_pre_nms(int): 训练模式时，nms前按分数取的bbox数量
        n_train_post_nms(int): 训练模式时，nms后保留的bbox数量
        n_test_pre_nms(int): 测试模式时，nms前按分数取的bbox数量
        n_test_post_nms(int): 测试模式时，nms后保留的bbox数量
        min_size(int): bbox的最小的尺寸
    '''
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
        print("self.n_train_pre_nms:", n_train_pre_nms)

    def __call__(self, loc, score, anchor, img_size, scale):
        '''

        :param loc: 对anchor的预测边界框偏移和比例
        :param score: anchor是正例的概率
        :param anchor: 通过枚举得到的所有anchor的坐标
        :param img_size: 图片大小
        :param scale: 宽高缩放倍数
        :return:
        '''
        print(self.n_train_pre_nms)
        if opt.mode == 'train':
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        # 通过bbox将anchor变换成proposal
        roi = loc2bbox(anchor, loc)
        # 将roi的坐标裁剪在img范围内
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])
        # 将高度或者宽度小于阈值的预测的bbox删除
        min_size = min(scale) * self.min_size
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >=min_size) & (ws >= min_size))[0]
        roi = roi[keep]
        score = score[keep]
        # (proposal, score)对按分数降序排序
        order = score.ravel().argsort()[::-1]
        # 取前n_pre_nms的(proposal, score)
        print(n_pre_nms)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        # 使用NMS(threshold=0.7)

        print(roi.shape, score.shape)

        print(keep.shape)

        exit(0)



