import numpy
import argparse
import simplejson as json
import pdb
import cv2 as cv
import os
from os import path
from tqdm import tqdm

img_root = '/home-ex/tclhk/chenww/G_EDAOI/'
img_saved_path ='/home-ex/tclhk/chenww/G_EDAOI_ROI_NEW/'

# img_root = '/home-ex/tclhk/chenww/M_EDMAI/'
# img_saved_path = '/home-ex/tclhk/chenww/M_EDMAI_ROI_NEW/'

# json_f = '/home-ex/tclhk/eric/T4_update/Datasets/Mask/VOC2007/mask.json'
json_f = '/home-ex/tclhk/eric/T4_update/Datasets/Glass/VOC2007/glass.json'

#"[434, 397, 517, 465], 0.9997506737709045]

def process_img(img,x1,y1,x2,y2):
    img_rows=img.shape[0] 
    img_cols=img.shape[1]

    length = x2 - x1
    width = y2 - y1
    center = [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 2)]

    extend = round(max(length, width) / 2 + max(length, width) * 0.15)
    new_x1, new_x2, new_y1, new_y2 = center[0] - extend, center[0] + extend, center[1] - extend, center[1] + extend  # try use lambda ,map

    if new_x1 < 0 or new_y1 < 0 or new_x2 > img_cols or new_y2 > img_rows:
        # print('ERROR:', filename)
        if new_x1 < 0:
            new_x1 = 0
        if new_y1 < 0:
            new_y1 = 0
        if new_x2 > img_cols:
            new_x2 = img_cols
        if new_y2 > img_rows:
            new_y2 = img_rows
    return new_x1,new_y1,new_x2,new_y2


            
# with open('./Mask_aug6x_cas_r50_dcn_top1.json') as f:
with open(json_f) as f:
    impath_class_box_dict = json.load(f)
    for path, mess in tqdm(impath_class_box_dict.items()): # [['EMDFBM', [499, 383, 696, 430], 0.8895947337150574]]
    # for path, mess in impath_class_box_dict.items():

        if mess==[]: # 没有检测框
            continue

        # for mask
        # img_name = path.split('/')[-1]
        # true_class = mess[0][0]

        # for glass
        img_name = path.split('/')[-1]
        # true_class = path.split('/')[-2]    # 这里应该没有True class
        # pdb.set_trace()
        xmin=mess[0][0]
        ymin=mess[0][1]
        xmax=mess[0][2]
        ymax=mess[0][3]

        # pdb.set_trace()
        img = cv.imread(os.path.join(img_root,img_name))    # 这里应该没有True class
        
        try:
            img.shape
        except:
            print('fail to read xxx.jpg')

        if 1:
            xmin, ymin, xmax, ymax= process_img(img,xmin,ymin,xmax,ymax)


        bbox = img[ymin:ymax, xmin:xmax]
        # save_path = os.path.join(img_saved_path)
        if not os.path.exists(img_saved_path):
            os.makedirs(img_saved_path)
        # print(save_path)
        cv.imwrite(os.path.join(img_saved_path,img_name),bbox)


