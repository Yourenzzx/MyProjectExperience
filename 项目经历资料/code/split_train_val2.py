import os
import os.path as osp
import random
from tqdm import tqdm
import shutil

"""
xml路径与img路径分开
"""
root = '/home-ex/tclhk/chenww/datasets/t2/d10/0309_0311/all/insidePanel/'
xml_path = '/home-ex/tclhk/chenww/datasets/t2/d10/0309_0311/all/xml/'
output_path = '/home-ex/tclhk/chenww/datasets/t2/d10/0309_0311/temp/'
# if osp.exists(output_path):
#     shutil.rmtree(output_path)
# os.makedirs(output_path)
ratio = 0.7

train_save_root = osp.join(output_path,'train')
val_save_root = osp.join(output_path,'val')
if not osp.exists(train_save_root):
    os.makedirs(train_save_root)
if not osp.exists(val_save_root):
    os.makedirs(val_save_root)


# get xml_list and img_list

gt_cls_list = os.listdir(root)
for gt_cls in gt_cls_list:

    path = osp.join(root, gt_cls)

    img_list = []
    for img_f in os.listdir(path):
        img_list.append(img_f)
        # xml_list.append(img_f.split('.jpg')[0]+'.xml')

    random.shuffle(img_list)
    l = len(img_list)

    train_save_path = osp.join(train_save_root, gt_cls)
    if not osp.exists(train_save_path): os.makedirs(train_save_path)
    val_save_path = osp.join(val_save_root, gt_cls)
    if not osp.exists(val_save_path): os.makedirs(val_save_path)

    # if(gt_cls == 'TSFAS'):
    #     print(len(img_list))
    #     assert False
    for idx, img_f in enumerate(img_list):
        xml_f = img_f.split('.jpg')[0] + '.xml'

        if idx <= int(l * ratio): # train
            shutil.copy(osp.join(root, gt_cls, img_f), train_save_path)
            # 如果是TSFAS,则不复制xml
            if gt_cls == 'TSFAS': continue
            try:
                shutil.copy(osp.join(xml_path, gt_cls, xml_f), train_save_path)
            except:
                print(gt_cls, xml_f)
            print('in {},idx {}'.format(gt_cls,idx))

        elif idx > int(l * ratio): # val
            shutil.copy(osp.join(root, gt_cls, img_f), val_save_path)
            # 如果是TSFAS,则不复制xml
            if gt_cls == 'TSFAS': continue
            try:
                shutil.copy(osp.join(xml_path, gt_cls, xml_f), val_save_path)
            except:
                print(gt_cls, xml_f)
            print('in {},idx {}'.format(gt_cls, idx))










