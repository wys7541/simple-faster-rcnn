import xml.dom.minidom as xmldom
import os


if __name__ == '__main__':
    path = r'C:\Users\Lenovo\Desktop\乱七八糟\PASCAL_VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations\000005.xml'
    dom = xmldom.parse(path)
    root = dom.documentElement
    bb = root.getElementsByTagName('folder')
    print(bb[0].firstChild.data)
    pose = root.getElementsByTagName('object')


