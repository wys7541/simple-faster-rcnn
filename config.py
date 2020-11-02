import warnings
import torch
import os

class DefaultConfig(object):
    '''
    项目默认配置参数类
    '''
    # data
    # 训练集和验证集存放路径
    trainval_data_root = 'C:\\Users\Lenovo\Desktop\PASCAL_VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    sample_file = os.path.join(trainval_data_root, 'sample.csv')
    use_gpu = False  # 是否使用gpu
    n_classes = 4   # 包括背景类的总类别数
    classes = ['bg', 'person', 'dog', 'cat']
    is_sampling = True  # 是否进行抽样
    total_samples = 90  # 总样本数
    width_size = 480     # 模型图片宽度
    height_size = 320    # 模型图片高度
    mode = 'train'     # 表示当前处于训练模式

    # training
    epoch = 2
    use_drop = False    # 是否使用dropout

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

