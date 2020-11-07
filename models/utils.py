import torch
import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    '''
    通过枚举纵横比和比例生成anchor
    :param base_size: 基本长度
    :param ratios: anchors的宽高比
    :param anchor_scales: 窗口的缩放倍数
    :return:
        ~numpy.ndarray:
        list(9, 4)->`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.
    '''
    py = base_size / 2.
    px = base_size / 2.
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            index = i * len(anchor_scales) + j
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1.0 / ratios[i])
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base

def normal_init(m, mean, stddev, truncated=False):
    '''
    网络结构参数初始化
    :param m: 网络名称
    :param mean: 均值
    :param stddev: 方差
    :param truncated: 是否保留几位小数
    :return:
    '''
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    '''
    通过移动枚举得到所有的anchor
    :param anchor_base: (numpy.ndarray 9*4)anchor_base
    :param feat_stride: (int)下采样倍数
    :param height: (int)特征图高度
    :param width: (int)特征图宽度
    :return: anchor（N, 4）
    '''
    anchors = np.zeros((height * width * len(anchor_base), 4), dtype=float)
    # 为每个feature map像素生成中心
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

def loc2bbox(src_bbox, loc):
    '''
    将rpn_locs转换成proposal bbox坐标
    x = (w_{a} * ctr_x_{p}) + ctr_x_{a}
    y = (h_{a} * ctr_x_{p}) + ctr_x_{a}
    h = np.exp(h_{p}) * h_{a}
    w = np.exp(w_{p}) * w_{a}
    and later convert to y1, x1, y2, x2 format
    :param src_bbox: anchor (N, 4) (ymin, xmin, ymax, xmax)
    :param loc: ([1, N, 4]) (dy, dx, dh, dw)
    :return:
        Args：ndarray numpy (N, 4)

    '''
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)  # 转换数组的数据类型
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dy * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w
    return dst_bbox

def bbox2loc(src_bbox, dst_bbox):
    '''
    给定边界框和目标边界框，此函数计算两者间的偏移和比例
    :param src_bbox: (N, 4) p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}
    :param dst_bbox: (N, 4) g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}
    :return:
        Bounding box offsets and scales: (N, 4)
    '''
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + height * 0.5
    ctr_x = src_bbox[:, 1] + width * 0.5

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + base_height * 0.5
    base_ctr_x = dst_bbox[:, 1] + base_width * 0.5

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def NMS(roi, score, iou_threshold):
    '''
    NMS
    :param boxes:
    :param scores:
    :param iou_threshold:
    :return:
    '''
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)    # 每一个检测框的面积
    order = score.argsort()[::-1]  # 按置信度降序排序
    keep = []   # 保留的框集合
    # 不断循环将得分最高的roi保留同时将小于threshhold的roi删除
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的那个
        # 计算得到相交区域的左上和右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IOU
        ovr = inter / (area[i] + area[order[1:]] - inter)
        # 保留小于threshhold的roi
        inds = np.where(ovr <=iou_threshold)[0]
        # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        order = order[inds + 1]
    return keep

"""
tools to convert specified type
"""
def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=False):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch.Tensor):
        return data.item()

def bbox_iou(rois, bbox):
    '''计算各个roi与gt_bbox间的iou'''
    if rois.shape[1] !=4 or bbox.shape[1] != 4:
        raise IndexError
    ious = np.empty((len(rois), len(bbox)), dtype=np.float32)
    for i, roi in enumerate(rois):
        ya1, xa1, ya2, xa2 = roi
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for j, box in enumerate(bbox):
            yb1, xb1, yb2, xb2 = box
            box_area = (ya2 - yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = iter_area / (anchor_area + box_area - iter_area)
            else:
                iou = 0.
            ious[i, j] = iou
    return ious


if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    anchor = enumerate_shifted_anchor(anchor_base, 16, 20, 30)
    print(anchor.shape)

