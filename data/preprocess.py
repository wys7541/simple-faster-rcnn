import os
from utils.config import opt
import csv



def data_sampling():
    '''
    抽取小样本的voc数据集用于实验
    :return: 返回生成的文件路径
    '''
    xmlfilepath = opt.trainval_data_root + '/Annotations'
    txtfilepath = opt.trainval_data_root + '/ImageSets/Main'
    imgfilepath = opt.trainval_data_root + '/JPEGImages'
    sample_file = opt.sample_file
    # 逐行写入 序号 类别 img xml
    with open(sample_file, 'w', newline='') as f:
        writer = csv.writer(f)
        cnt = 0
        # 各个类别依次遍历
        for i in range(0, len(opt.classes)):
            class_name = opt.classes[i]
            class_txt_name = class_name + '_trainval.txt'
            class_txt_path = os.path.join(txtfilepath, class_txt_name)
            # 每个类别取total_samples/n_classes个样本
            with open(class_txt_path, 'r', encoding='utf-8') as txt:
                line = txt.readline()
                while line:
                    line = line.strip()
                    id, flag = line.split()
                    if flag == '1':
                        img_path = os.path.join(imgfilepath, id+'.jpg')
                        xml_path = os.path.join(xmlfilepath, id+'.xml')
                        # print(id, flag)
                        writer.writerow([cnt, class_name, img_path, xml_path])
                        cnt += 1
                    if cnt >= (i + 1) * (opt.total_samples / opt.n_classes):
                        break
                    line = txt.readline()
                txt.close()
        f.close()
    return sample_file




if __name__ == '__main__':
    data_sampling()