from torch.utils.data import DataLoader, Dataset
from utils.config import opt
import csv
from data import util

class Transform(object):
    '''
    数据变换， 得到图片变换后的img，bbox
    '''
    def __init__(self, width_size=480, height_size=320):
        self.width_size = width_size
        self.height_size = height_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = util.resize_img(img, self.width_size, self.height_size)
        _, o_H, o_W = img.shape
        scale = [o_H / H, o_W / W]
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        return img, bbox, label, scale


class MyDataset(Dataset):
    def __init__(self, opt):
        super(MyDataset, self).__init__()
        self.opt = opt
        fh = open(self.opt.sample_file, 'r')
        reader = csv.reader(fh)
        dataset = []
        for row in reader:
            # row -> id, class, img_path, xml_path
            dataset.append((int(row[0]), opt.classes.index(row[1]), row[2], row[3]))
        self.dataset = dataset
        self.tsf = Transform(opt.width_size, opt.height_size)

    def __getitem__(self, index):
        '''
        按照索引读取每个元素的具体内容
        :param index: 索引
        :return:
            ~numpy.ndarray：img, bbox, label
        '''
        row = self.dataset[index]
        ori_img = util.read_image(row[2])
        bbox, label = util.read_voc_xml(row[3], self.opt.classes)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img, bbox, label, scale

    def __len__(self):
        '''返回数据集的长度'''
        return len(self.dataset)

if __name__ == '__main__':
    train_data = MyDataset(opt)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers = 2)
    for i, data in enumerate(train_loader):
        # print(data)
        print(data[0].size())
        print(data[1].size())
        print(data[2].size())
        print(len(data[3]))
        exit()
    pass


