import warnings
import torch

class DefaultConfig(object):
    # 训练集存放路径
    train_data_root = r'C:\Users\Lenovo\Desktop\乱七八糟\PASCAL_VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
    use_gpu = False  # user GPU or not

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
            if not k.starswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()

