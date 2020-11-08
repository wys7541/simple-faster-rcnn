import warnings
import torch
import os

class DefaultConfig(object):
    '''
    项目默认配置参数类
    '''
    # data
    # 训练集和验证集存放路径
    trainval_data_root = '/data/Matt/MyTestModel/simple-faster-rcnn/data/voc/VOCdevkit/VOC2007'
    sample_file = os.path.join(trainval_data_root, 'sample.csv')
    use_gpu = False     # 是否使用gpu
    n_classes = 3       # 不包括背景类
    classes = ['person', 'dog', 'cat']
    is_sampling = True  # 是否进行抽样
    total_samples = 90  # 总样本数
    width_size = 480    # 模型图片宽度
    height_size = 320   # 模型图片高度
    mode = 'train'      # 表示当前处于训练模式

    # training
    epoch = 1
    use_drop = False    # 是否使用dropout

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3
    use_adam = False    # 是否使用Adam optimizer

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # model
    load_path = None

    # caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain = False
    # caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
    caffe_pretrain_path = None

    def _parse(self, kwargs):
        """
           根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
                setattr(self, k, v)
        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()

