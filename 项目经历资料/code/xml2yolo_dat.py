import os
import xml.dom.minidom as minidom
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import pdb

from random import shuffle


'''
function:  从每张图片对应的xml整理成一个.dat文件给检测模型
文件的格式为：[图片绝对路径 cls 1 cx cy fw fh  1 cx cy fw fh  ...]
'''

img_root = '/home-ex/tclhk/chenww/datasets/t2/temp2/train/'
output_dat_path = '/home-ex/tclhk/chenww/t2/yolo_v3_x/config/d10/temp/'
train_or_val = 'train' # 生成'train.dat或生成val.dat'

def read_xml(xml_filename):
    dom = minidom.parse(xml_filename)
    root = dom.documentElement
    assert (len(root.getElementsByTagName('filename')) == 1)
    assert (len(root.getElementsByTagName('size')) == 1)

    for filename in root.getElementsByTagName('filename'):
        filename = filename.firstChild.data

    # for c in root.getElementsByTagName('folder'):
    #     cls = c.firstChild.data

    # for size in root.getElementsByTagName('size'):
    #     width = size.getElementsByTagName('width')[0].firstChild.data
    #     height = size.getElementsByTagName('height')[0].firstChild.data
    #     depth = size.getElementsByTagName('depth')[0].firstChild.data
    #     # print(width, height, depth)

    label_name_list = []
    for i, label_name in enumerate(root.getElementsByTagName('name')):
        ln = label_name.firstChild.data
        label_name_list.append(ln)

    bboxes = []
    for bndbox in root.getElementsByTagName('bndbox'):
        xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.data
        ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.data
        xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.data
        ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.data
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        bboxes.append((xmin, ymin, xmax, ymax))
    # return filename, bboxes, cls
    return  bboxes, label_name_list


if not os.path.exists(output_dat_path):
    os.makedirs(output_dat_path)

class_list = []
info_list = []
tsfas_list = []

# class list
for class_name in os.listdir(img_root):
    if os.path.isdir(os.path.join(img_root, class_name)):
        class_list.append(class_name)
print("total_classes:", class_list)

for i, class_name in enumerate(class_list):
    path = os.path.join(img_root, class_name)
    all_file = os.listdir(path)

    img_list = []
    for f in all_file:
        # if f.endswith('jpg'):
        if f.endswith('JPG'):
            img_list.append(f)

    for img_file in img_list[:]:
        xml = img_file[:-3] + 'xml'

        xml_filename_path = os.path.join(img_root,class_name, xml)
        img_file_path = os.path.join(img_root,class_name, img_file)

        #-----------------------filter-------------------------
        # if class_name == 'TSFAS':
        if class_name == 'TFOL0':
            tsfas_list.append((img_file_path))
            continue # 遇到TSFAS 返回，不读取标注文件
        #-----------------------filter-------------------------

        try:  # 防止xml为空或者没有xml这个文件，会发生错误
            # filename, bboxes, cls = read_xml(xml_filename_path)
            bboxes, label_name_list = read_xml(xml_filename_path)
        except:
            print('bbox in xml is none:', xml)
            continue

        # -----------------------filter-------------------------
        # when I used label_img , I named the box with the unreasonable category "out"
        # ignoring image with name "out"
        if 'out' in label_name_list and class_name == 'TCPIA':
            print('image out:', fi)
            continue
        # -----------------------filter-------------------------

        fi = os.path.join(img_root, class_name, img_file)
        img = cv2.imread(img_file_path)
        height, width, _ = img.shape
        assert img is not None, img_file_path

        output_txt = ''
        b = ''
        for xmin, ymin, xmax, ymax in bboxes:
            w = xmax - xmin
            h = ymax - ymin

            # -----------------------filter-------------------------

            if class_name == 'TCSAD' or class_name == 'TSILR':
                if w < 50 and h < 50:
                    continue
            # -----------------------filter-------------------------

            fw = (w + 0.0) / width
            fh = (h + 0.0) / height
            cx = (xmin + 0.0) / width + fw / 2
            cy = (ymin + 0.0) / height + fh / 2
            # b += '0 {} {} {} {} {} '.format(cx, cy, fw, fh, class_name)
            b += '1.0 {} {} {} {} '.format(cx, cy, fw, fh)
        info_list.append((fi, class_name, b))


# shuffle(info_list)
num_file = len(info_list) + len(tsfas_list)
print("len of tsfas:", len(tsfas_list))
print("len of others:", len(info_list))
print("num_file:", num_file)

if train_or_val == 'train':
    output_dat_path = output_dat_path + 'train.dat'
else:
    output_dat_path = output_dat_path + 'val.dat'

with open(output_dat_path, 'w') as f:
    for img_f, cls, box in info_list:
            f.write('{} {} {}\n'.format(img_f, cls, box))
    for img_f in tsfas_list:
        f.write('{} TFOL0\n'.format(img_f))

print('done')