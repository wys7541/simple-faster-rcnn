import torch
import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    '''
    通过枚举纵横比和比例生成bbox
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
    # 不断循环将得分最高的bbox保留同时将小于threshhold的bbox删除
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
        # 保留小于threshhold的bbox
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



if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    anchor = enumerate_shifted_anchor(anchor_base, 16, 20, 30)
    print(anchor.shape)

