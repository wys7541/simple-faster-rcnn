import torch
import numpy as np
from models.utils import *

class AnchorTargetCreator(object):
    '''
    将gt_box指派给anchors
    Args:
        n_sample(int): 产生的sample regions数量
        pos_iou_thresh(float): iou大于pos_iou_thresh的anchor设置为postive
        neg_iou_thresh(float): iou小于neg_iou_thresh的anchor设置为negtive
        pos_ratio: postive anchor的占比
    '''
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        '''
        给sample anchor指派loc和label,并且将其余的anchor loc为0，label为-1
        :param bbox: bbox 的坐标 (R, 4)
        :param anchor: anchors的坐标 (S, 4)
        :param img_size: (tuple of ints) H, W
        :return:
            Args:(array, array)
                loc: (S, 4)
                label:(S,)
        '''
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = self._get_inside_index(anchor, img_H, img_W)
        # 取坐标全在图片内部的anchor
        anchor = anchor[inside_index]
        # argmax_ious代表各个roi与哪个bbox的iou最大
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # 计算边界框回归目标
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # label,loc 映射到原始的anchor的顺序，此时argmax_ious、inside都是索引组成的array
        label = self._unmap(label, n_anchor, inside_index, fill=-1)
        loc = self._unmap(loc, n_anchor, inside_index, fill=0)
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        '''
        给n_sample个anchor指定最合适的label
        label: 1 postive; label: 0 negtive; label: -1 dont care
        :return
            argmax_ious(N,): 各个roi与哪个bbox的iou最大
            label：(N,) 其中有n_sample个非-1值，表示anchor是negtive、postive or dont care
        '''
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        # 分配负标签（0）给max_iou小于负阈值的所有anchor boxes
        label[max_ious < self.neg_iou_thresh] = 0
        # 分配正标签（1）给与ground-truth box的IoU重叠最大的anchor boxes
        label[gt_argmax_ious] = 1
        # 分配正标签（1）给max_iou大于positive阈值的anchor boxes
        label[max_ious >= self.pos_iou_thresh] = 1

        # 从正标签中随机采样n_pos个样本，忽略（-1）剩余的样本。在一些情况下，得到少于n_pos个样本，
        # 此时随机采样（n_sample - n_pos）个负样本（0），忽略剩余的anchor boxes
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index,size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        '''anchar与gt_boxes的ious
            anchor(N, 4), bbox(R, 4), (N,)
            :return
                argmax_ious: (N,) ious的每一行的最大值的所在列数组成的array
                max_ious：(N,) ious的每行最大值组成的array
                gt_argmax_ious：(K,) rois中与某个bbox有最大iou的索引组成的array
        '''
        ious = bbox_iou(anchor, bbox)
        # argmax_ious(N,) ious的每一行的最大值的所在列组成的array
        argmax_ious = ious.argmax(axis=1)
        # max_ious(N,) ious的每行最大值组成的array
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        # gt_argmax_ious(R,) ious中分别与bbox的iou最大的R个roi的索引
        gt_argmax_ious = ious.argmax(axis=0)
        # np.arange(ious.shape[1]) 使用这句是因为gt——argmax_ious的第i个值是rois第i列的最大值
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


    def _get_inside_index(self, anchor, H, W):
        '''计算得到完全处在图片内部的anchor'''
        index_inside = np.where(
            (anchor[:, 0] > 0) &
            (anchor[:, 1] > 0) &
            (anchor[:, 2] < H) &
            (anchor[:, 3] < W))[0]
        return index_inside

    def _unmap(self, data, count, index, fill=0):
        ''''''
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=data.dtype)
            ret.fill(fill)
            ret[index] = data

        else:
            ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[index, :] = data
        return ret




class ProposalTargetCreator(object):
    '''
    设定proposal目标,将gt_bbox指派给给定的RoI
        Args:
            n_sample: roi中采样的样本数目
            pos_ratio: n_samples中正样本的比例
            pos_iou_thresh: 设置为正样本region proposal与gt_bbox的最小iou值
            neg_iou_thresh_hi: 设置为负样本背景的iou
            neg_iou_thresh_lo:
    '''
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):

        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        '''
        给sampled proposals指派ground truth
        :param roi: (N, 4)待抽样的rois
        :param bbox: (K, 4)gt_bbox 的坐标
        :param label: (K,)gt_bbox 的标签
        :param loc_normalize_mean:(4,) bbox坐标的平均值
        :param loc_normalize_std: (4,)bbox的标准差
        :return:
            Args：(array, array, array)
            sample_roi(S, 4): 抽样出的roi
            gt_roi_loc(S, 4): 被抽样roi的gt_bbox
            gt_roi_label(S,): 分配给sample_roi的label
        '''
        label = np.asarray([i for i in range(len(label))])
        n_bbox, _ = bbox.shape
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)

        # 计算各个roi与gt_bbox间的iou(N, k)
        iou = bbox_iou(roi, bbox)
        # 找到与每个region proposal具有较高IoU的ground truth，并且找到最大的IoU
        gt_assignment = iou.argmax(axis=1)  # (N, )
        max_iou = iou.max(axis=1)   # (N, )
        # 为每个proposal分配标签, 这里暂不考虑background，后面将非postive统一为negtive
        gt_roi_label = label[gt_assignment] # (N, )

        # 挑选foreground rois， iou >= pos_iou_thresh
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            # 当样本大于pos_roi_per_image则再进行抽样
            pos_index = np.random.choice(pos_index, size=pos_roi_per_image, replace=False)
        # 挑选background RoIs，iou between neg_iou_thresh_lo and neg_iou_thresh_hi
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        # 合并成postive negtive
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_image:] = 0    # negative labels --> 0
        sample_roi = roi[keep_index]

        # 计算偏移量和比例，以将采样的ROI与GTs匹配
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) /
                      np.array(loc_normalize_std, np.float32))
        print("ProposalTargetCreator return:")
        print("sample_roi.shape:", sample_roi.shape)
        print("gt_roi_loc.shape:", gt_roi_loc.shape)
        print("gt_roi_label.shape:", gt_roi_label.shape)
        return sample_roi, gt_roi_loc, gt_roi_label
