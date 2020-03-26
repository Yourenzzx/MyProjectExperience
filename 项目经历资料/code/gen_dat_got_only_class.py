import os
import xml.dom.minidom as minidom
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import pdb

from random import shuffle


img_root = '/home-ex/tclhk/chenww/datasets/t2/d10/2253-D10-0112~0212/insidePanel/'
output_dat_path = '/home-ex/tclhk/chenww/datasets/t2/d10/2253-D10-0112~0212/'
output_dat_name = 'insidePanel.dat'
list_of_12code = ['TCOTS', 'TCPIA', 'TCPOA', 'TCSAD', 'TGGS0', 'TPDPS', 'TSDFS', 'TSILR', 'TTFBG', 'TTP3G', 'TTSPG', 'TSFAS']

if not os.path.exists(output_dat_path):
    os.makedirs(output_dat_path)

class_list = []
info_list = []


# class list
for class_name in os.listdir(img_root):
    if os.path.isdir(os.path.join(img_root, class_name)):
        class_list.append(class_name)
print("total_classes:", class_list)

"""
# folder structure: root/gt_cls/wrg_cls/img...
for i, class_name in enumerate(class_list):
    if class_name not in list_of_12code: continue

    for w_cls in os.listdir(os.path.join(img_root, class_name)):
        path = os.path.join(img_root, class_name, w_cls)
        for img_file in os.listdir(path):
            if img_file.endswith('.xml'): continue
            fi = os.path.join(img_root, class_name, w_cls, img_file)
            info_list.append((fi, class_name))
"""

# folder structure: root/gt_cls/img...
for i, class_name in enumerate(class_list):
    # if class_name not in list_of_12code: continue

    path = os.path.join(img_root, class_name)
    for img_file in os.listdir(path):
        if img_file.endswith('.xml'): continue
        fi = os.path.join(img_root, class_name, img_file)
        info_list.append((fi, class_name))

# shuffle(info_list)
print("len of others:", len(info_list))

output_dat_path = output_dat_path + output_dat_name

with open(output_dat_path, 'w') as f:
    for img_f, cls in info_list:
        f.write('{} {} \n'.format(img_f, cls))

print('done')