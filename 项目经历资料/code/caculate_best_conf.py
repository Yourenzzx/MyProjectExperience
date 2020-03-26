import json
import os
from collections import defaultdict
import numpy as np

# 由之前的combine_test结果，和val.dat计算最佳置信度阈值
final_result_path = '/home-ex/tclhk/chenww/t2/models/combine/0107_TCPIA500_finetune_from_x_det_ep48_conf0.5_nms0.0_cls_ep30/final_result.json'
dat_file_path = '/home-ex/tclhk/chenww/t2/yolo_v3_x/config/v3_TCPIA550/val.dat'

with open(dat_file_path) as g:
    gt_data = g.readlines()
gt_dict = {}

for da in gt_data:
    gt_dict[da.strip().split()[0]] = da.strip().split()[1]

with open(final_result_path) as f:
    model_out = json.load(f)

dis = defaultdict(int)
for name, mess_dict in model_out.items():
    dis[mess_dict['label']] += 1
class_list = list(dis.keys())
class_to_id = {name: idx for idx, name in enumerate(class_list)}
print(class_to_id)
conf_ther_list = np.linspace(0, 1, 21)
# Mat = np.zeros((len(class_list), len(class_list)), np.int32)
total_sample = len(model_out)
print('total number {}'.format(total_sample))
print('{:<7}{:<7}{:<7}{:<7}'.format('conf', 'acc', 'ratio', 'a&r'))
for conf_ther in conf_ther_list:
    Mat = np.zeros((len(class_list), len(class_list)), np.int32)
    unconfirm_number = 0
    for name, mess_dict in model_out.items():
        gt_name = gt_dict[name]
        pre_name = mess_dict['label']
        gt_id = class_to_id[gt_name]
        pre_id = class_to_id[pre_name]
        conf = mess_dict['conf']
        if conf >= conf_ther or pre_name == 'TSFAS':
            Mat[gt_id][pre_id] += 1
        else:
            unconfirm_number += 1
    colsum = np.sum(Mat, axis=0).tolist()
    # rowsum = np.sum(Mat, axis=1).tolist()
    total = np.sum(colsum, axis=0)
    diag = np.trace(Mat)
    total_acc = diag / total

    confirm_ratio = 1 - unconfirm_number / total_sample
    print('{:<7.4f}{:<7.4f}{:<7.4f}{:<7.4f}'.format(conf_ther, total_acc, confirm_ratio, (total_acc * confirm_ratio)))



