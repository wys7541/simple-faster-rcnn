import torch
import torch.nn as nn
from utils.config import opt
from torchvision.models import vgg16
from models import FasterRCNN
from models import RegionProposalNetwork
from models import VGG16RoIHead

def decom_vgg16():
    '''用vgg16特征提取, 返回特征提取器和分类器'''
    if opt.caffe_pretrain:
       model = vgg16(pretrained=False)
       if not opt.load_path:
            model.load_state_dict(torch.load(opt.caffe_pretrain_path))
    else:
        # model = vgg16(not opt.load_path)    # pretrained = True
        model = vgg16(False)
    features = list(model.features)[:30]
    classifier = model.classifier
    del classifier[6]
    # 是否使用dropout
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    # # freeze top4 conv
    # for layer in features[:10]:
    #     for p in layer.parameters():
    #         p.requires_grad = False
    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    '''基于VGG-16的Faster R-CNN
    Args:
        n_classes(int): 包括背景类的总类别数
        ratios(list of float): anchor的宽高比例
        anchor_scales(list of number): 基础长度
    '''
    feat_stride = 16    # extractor输出下采样16px

    def __init__(self,
                 n_classes=opt.n_classes,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios= ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )
        head = VGG16RoIHead(
            n_class = n_classes + 1,    # 包括bg的总类别数
            roi_size = 7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head
        )
