import numpy as np
from PIL import Image
import random
from skimage import transform as sktsf
import xml.etree.ElementTree as ET

def read_image(path, dtype=np.float32, color=True):
    '''
    读取图片文件
    :param path: 图片文件路径
    :param dtype: 数组元素类型
    :param color: True 彩色图片， False 灰度图片
    :return:
        ~numpy.ndarray： 一张图片
    '''
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()
    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # reshape (H, W, C) -> (C, H, W)
        return img.transpose(2, 0, 1)

def read_voc_xml(path, classes):
    '''
    读取xml格式文件
    :param path: xml路径
    :param classes: 标签名称
    :return:
        ~numpy.ndarray： float32 bbox, int32 label
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    bbox = list()
    labels = list()
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        if object_name in classes:
            labels.append(classes.index(object_name))
            bbox.append([Ymin, Xmin, Ymax, Xmax])
    bbox = np.stack(bbox).astype(np.float32)
    labels = np.stack(labels).astype(np.int32)
    return bbox, labels

def resize_img(img, width_size, height_size):
    '''
    改变图片的尺寸
    :param img: numpy.ndarray img
    :param width_size:  int
    :param height_size:  int
    :return:
        ~numpy.ndarray resizeed_img
    '''
    C, H, W = img.shape
    return sktsf.resize(img, (C, height_size, width_size))

def resize_bbox(bbox, in_size, out_size):
    '''
    根据图片尺寸改变bbox的尺寸
    :param bbox: ~numpy.ndarray
    :param in_size: tuple(ori_h, ori_w)
    :param out_size: tuple(h, w)
    :return: bbox
    '''
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / out_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox