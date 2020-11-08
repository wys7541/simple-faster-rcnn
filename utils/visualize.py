import os
import cv2
import xml.etree.ElementTree as ET
import random
import numpy as np
import matplotlib.pyplot as plt

def ploy_xml_img(xml_path, img_path, save=False, save_path=None):
    '''
    将voc数据集的图片和标注文件结合展示
    :param self:
    :param xml_path: xml路径
    :param img_path: img路径
    :param save: 是否保存
    :param save_path: 保存路径
    :return: null
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img = cv2.imread(img_path)
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(img, (Xmin, Ymin), (Xmax, Ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, object_name, (Xmin, Ymin-7), font, 0.5, color, 2)
    cv2.imshow('01', img)
    cv2.waitKey(0)
    if save == True and save_path != None:
        cv2.imwrite(save_path, img)

def loss_plot(losses):
    '''
    loss折线图
    :param losses: (list of float)
    :return:
    '''
    x = np.arange(len(losses))
    y = np.ndarray(losses)
    plt.plot(x, y)

    plt.title('Loss Chart')
    plt.xlabel('cnt')
    plt.ylabel('loss')
    plt.savefig('loss.jpg')
    plt.show()




if __name__ == '__main__':
    xml_path = r'C:\Users\Lenovo\Desktop\PASCAL_VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations\000005.xml'
    img_path = r'C:\Users\Lenovo\Desktop\PASCAL_VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000005.jpg'
    ploy_xml_img(xml_path, img_path)