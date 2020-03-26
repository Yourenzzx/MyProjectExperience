from __future__ import division
import csv
from collections import Counter
import os
import cv2
import pdb
import numpy as np

list_of_12code = ['TCOTS', 'TCPIA', 'TCPOA', 'TCSAD', 'TGGS0', 'TPDPS', 'TSDFS', 'TSILR', 'TTFBG', 'TTP3G', 'TTSPG', 'TSFAS']

"""
输入：csv文件的绝对路径 && 类别所在列的索引
输出：各个类别的数目统计
"""
def count_cls_num(csv_file, cls_idx_in_csv):
    cls = []
    with open(csv_file) as f:
        csv_f = csv.reader(f)
        for idx, row in enumerate(csv_f):
            if idx != 0:
                cls.append(row[cls_idx_in_csv])
        cls_counter = Counter(cls)
        return cls_counter
"""
输入：csv文件的绝对路径
输出：包含每一行信息的列表 + 类别的统计（仅仅限定在d13-2250的12个code中）
"""
def get_csv_info(csv_file):
    csv_info_list = []
    cls_list = []
    with open(csv_file) as f:
        csv_f = csv.reader(f)
        for idx, row in enumerate(csv_f):
            if idx == 0: continue
            csv_info_list.append(row)
            if row[10] in list_of_12code:
                cls_list.append(row[10]) # row[10] = gt_cls
    cls_list = list(set(cls_list))

    return csv_info_list, cls_list


# cls_dict = { name: id}
"""
输入：包含csv文件每一行信息的列表,类别列表, 分类的阈值
输出：混淆矩阵、低于分类阈值的图片数目统计
"""
def get_confusion_mat(csv_info_list, cls_list, conf_thres):
    id_list = [i for i in range(len(cls_list))]
    cls_id_dict = dict(zip(cls_list, id_list)) # from cls_list : id_list generate cls_id_dict
    confusion_mat = np.zeros((len(cls_id_dict), len(cls_id_dict)), np.int32)
    unconfirm_number = 0
    pred_turnon_list = []
    for csv_info in csv_info_list:
        score = float(csv_info[5])
        gt_cls = csv_info[10]
        pred_cls = csv_info[4]

        if score == 2: continue # 非模板

        if score == 1 and pred_cls == gt_cls: # 系统判turn on 且 pred_cls == gt_cls
            pred_turnon_list.append(pred_cls)

        # gt_cls = csv_info[10]
        if gt_cls not in list_of_12code : continue
        gt_id = cls_id_dict[gt_cls]
        pred_cls = csv_info[4]
        pred_id = cls_id_dict[pred_cls]

        if score > conf_thres :
            confusion_mat[gt_id][pred_id] += 1
        else:
            unconfirm_number += 1

    temp_counter = Counter(pred_turnon_list)
    print("pred_turnon && gt_cls == pred_cls:", temp_counter)

    return confusion_mat, unconfirm_number

"""
输入：包含csv文件每一行信息的列表,类别列表, 分类的阈值列表
输出：打印出不同阈值下的准确率，覆盖率
"""
def get_acc_cover_ratio(csv_info_list, cls_list, conf_thre_list):
    id_list = [i for i in range(len(cls_list))]
    cls_id_dict = dict(zip(cls_list, id_list))  # from cls_list : id_list generate cls_id_dict
    unconfirm_number = 0
    pred_turnon_list = []
    print('thres     acc      ratio  acc*ratio')
    for i, conf_thre in enumerate(conf_thre_list):
        confusion_mat = np.zeros((len(cls_id_dict), len(cls_id_dict)), np.int32)
        for csv_info in csv_info_list:
            score = float(csv_info[5])
            gt_cls = csv_info[10]
            pred_cls = csv_info[4]

            if score == 2 or score == 1: continue # 非模板及turn on
            if gt_cls not in list_of_12code : continue

            gt_id = cls_id_dict[gt_cls]
            pred_cls = csv_info[4]
            pred_id = cls_id_dict[pred_cls]

            if score > conf_thre :
                confusion_mat[gt_id][pred_id] += 1
            else:
                unconfirm_number += 1

        colsum = np.sum(confusion_mat, axis=0).tolist()
        total = np.sum(colsum, axis=0)
        diag = np.trace(confusion_mat)
        total_acc = diag / total

        confirm_ratio = total / len(csv_info_list)

        print('{:<7.4f}  {:<7.4f}  {:<7.4f}  {:<7.4f}'.format(conf_thre, total_acc, confirm_ratio, \
                                                    (total_acc * confirm_ratio)))

    return confusion_mat, unconfirm_number

def print_confusion_mat(confusion_mat, cls_list):
    print(('class  ' + '{:<7}' * len(confusion_mat)).format(*cls_list))
    for name, dat in zip(cls_list, confusion_mat):
        prstr = ''
        prstr += '{:<7}'.format(name)
        prstr += ('{:<7}' * len(cls_list)).format(*dat)
        print(prstr)
    colsum = np.sum(confusion_mat, axis=0).tolist()
    rowsum = np.sum(confusion_mat, axis=1).tolist()
    total = np.sum(colsum, axis=0)
    diag = np.trace(confusion_mat)
    recall = np.diagonal(confusion_mat) / rowsum
    precision = np.diagonal(confusion_mat) / colsum
    print('class   recall precition')
    for name, rec, pre in zip(cls_list, recall, precision):
        print('{:<8}{:<7.4f}{:<7.4f}'.format(name, rec, pre))
    print('total acc {}'.format(diag / total))

def print_confusion_mat_from_csv(csv_file):
    csv_info_list, cls_list = get_csv_info(csv_file)
    # confusion_mat, unconfirm_number = get_confusion_mat(csv_info_list, cls_list, 0)
    get_acc_cover_ratio(csv_info_list, cls_list, [50,55,60,65,70,75,80,85])
    # confusion_mat, unconfirm_number = get_confusion_mat(csv_info_list, cls_list, 0)
    # print_confusion_mat(confusion_mat, cls_list)


if __name__ == '__main__':
    # TPDPS/TCOTS/TTFBG needed to turn on ,so I can control threshold to maximum the accuracy of 3 codes
    # 优化思路：调整不同的阈值，提高这三类需要turn on的code的准确率
    csv_f = "/home-ex/tclhk/chenww/t2/online_excel/2250-D13-20200216_17_18.csv"
    # csv_f = "/home-ex/tclhk/chenww/t2/online_excel/2250-D13-20200217.csv"
    print_confusion_mat_from_csv(csv_f)
    # thres = 70
    # thres_list = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    # turn_on_list = ['TPDPS', 'TCOTS', 'TTFBG']
    #
    # # 将表格信息存储到List中
    # csv_info_list = get_csv_info(csv_f)
    #
    # for thres in thres_list:
    #     above_thres_info_list = []
    #     # 取出高于阈值的表格信息
    #     for csv_info in csv_info_list:
    #         score = float(csv_info[5])
    #         if score == 1 or score == 2: continue
    #         if score > thres:
    #             above_thres_info_list.append(csv_info)
    #
    #     right_clssify_turnon_code_info_list = []
    #     turn_on_code_info_list = []
    #     for above_thres_info in above_thres_info_list:
    #         gt_cls = above_thres_info[10]
    #         pred_cls = above_thres_info[4]
    #
    #         if gt_cls not in turn_on_list:
    #             continue
    #
    #         turn_on_code_info_list.append(above_thres_info)
    #
    #         if gt_cls == pred_cls:
    #             right_clssify_turnon_code_info_list.append(above_thres_info)
    #
    #     acc = len(right_clssify_turnon_code_info_list) / len(turn_on_code_info_list)
    #
    #     print("thres:", thres)
    #     print("acc:",acc)
    #     print('-------------')

    # 对于不同的阈值，算出表格的覆盖率及准确率
    # for thres in thres_list:
    #     above_thres_info_list = []
    #     for csv_info in csv_info_list:
    #         score = float(csv_info[5])
    #         if score == 1 or score == 2: continue
    #         if score > thres:
    #             above_thres_info_list.append(csv_info)
    #
    #     right_clssify_in_above_thres_list = []
    #     for above_thres_info in above_thres_info_list:
    #         gt_cls = above_thres_info[10]
    #         pred_cls = above_thres_info[4]
    #
    #         if gt_cls == pred_cls:
    #             right_clssify_in_above_thres_list.append(above_thres_info)
    #
    #     cover_ratio = len(above_thres_info_list) / len(csv_info_list)
    #     acc = len(right_clssify_in_above_thres_list) / len(above_thres_info_list)
    #
    #     print("thres:", thres)
    #     print("cover_ratio:",cover_ratio)
    #     print("acc:",acc)
    #     print('-------------')
