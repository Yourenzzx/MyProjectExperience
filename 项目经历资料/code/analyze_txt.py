from __future__ import division
import os
import os.path as osp
import sys
import argparse
import pdb
import numpy as np
import pdb
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_data(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    return lines

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=10)

    # parser.add_argument("--log_path", type=str, default='/home-ex/tclhk/chenww/t2/models/yolo_v3_x/d10/0320_data0309_0311_finetuneD13/log.txt')
    parser.add_argument("--log_path", type=str, default='/home-ex/tclhk/chenww/t2/models/yolo_v3_x/d10/temp/log.txt')
    parser.add_argument("--save_path", type=str, default='/home-ex/tclhk/chenww/t2/models/yolo_v3_x/d10/0320_data0309_0311_finetuneD13/analysis_log/')

    parser.add_argument("--ep_sample_iter", type=int, default=5)

    parser.add_argument("--show_loss", default=True, choices=['True', 'False'])
    parser.add_argument("--show_diff_conf_thresh", default=True, choices=['True', 'False'])
    parser.add_argument("--show_no_defect_image_acc", default=True, choices=['True', 'False'])
    parser.add_argument("--show_defect_image_acc", default=True, choices=['True', 'False'])
    parser.add_argument("--show_bbox_acc", default=True, choices=['True', 'False'])
    parser.add_argument("--show_bbox_recall", default=True, choices=['True', 'False'])
    args = parser.parse_args(argv)

    thres_list = [0.1, 0.3, 0.5, 0.7]
    # thres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    thres_num = len(thres_list)
    figureCol = 2
    figureRow = math.ceil(thres_num / 2) + 1  # 一行用来专门显示loss信息

    # get .dat lines
    lines = get_data(args.log_path)
    lines_len = len(lines)
    # acc
    no_defect_img_acc = []
    defect_img_acc = []
    bbox_acc = []
    bbox_recall = []

    ndacc= [0] * thres_num
    dacc = [0] * thres_num
    bacc = [0] * thres_num
    brcl = [0] * thres_num
    print(ndacc)

    # loss
    total_loss_list = []
    xy_loss_list = []
    wh_loss_list = []
    conf_loss_list = []

    # temp
    cnt = 0

    # 遍历每行
    for idx, line in enumerate(lines):
        if idx < 34:  continue  # 前34行有干扰信息
        if not line.strip().endswith(".pth"):  # 如果不是以yolov3_ckpt_x.pth为开始
            continue

        epoch = line.strip().split("_")[2].split(".")[0]
        # epoch = line.strip().split("_")[1].split(".")[0]
        try: epoch = int(epoch)
        except: continue

        #  指定的epoch区间
        if args.start_epoch > epoch or epoch > args.end_epoch or epoch == 0:
            continue

        #  =============================== get acc info ==================================
        for th_idx, i in enumerate(range(1, 5 * thres_num, 5)):
            data = lines[idx + i].strip().split()

            # print(epoch)
            if not epoch % args.ep_sample_iter == 0:  #  如果行数不为采样频数的整数倍
                ndacc[th_idx] += float(data[1])
                dacc[th_idx] += float(data[2])
                bacc[th_idx] += float(data[4])
                brcl[th_idx] += float(data[6])
                # pdb.set_trace()
            else:
                # pdb.set_trace()
                no_defect_img_acc.append(ndacc[th_idx] / (args.ep_sample_iter-1))
                defect_img_acc.append(dacc[th_idx] / (args.ep_sample_iter-1))
                bbox_acc.append(bacc[th_idx] / (args.ep_sample_iter-1))
                bbox_recall.append(brcl[th_idx] / (args.ep_sample_iter-1))
                ndacc[th_idx] = 0
                dacc[th_idx] = 0
                bacc[th_idx] = 0
                brcl[th_idx] = 0
        #  =============================== get acc info ==================================

        #  =============================== get loss info ==================================
        # i = idx + 21
        i = idx + thres_num * 5 + 1
        loss_info_cnt = 0

        loss = 0
        xy_loss = 0
        wh_loss = 0
        conf_loss = 0

        # 获取Loss信息，与batch_size的设定有关，行数不定
        while (i < lines_len and lines[i][0] == '['):
            loss += float(lines[i].strip().split()[6])
            xy_loss += float(lines[i].strip().split()[8])
            wh_loss += float(lines[i].strip().split()[10])
            conf_loss += float(lines[i].strip().split()[12])
            # lr = lines[i].strip().split()[14]

            i = i + 1
            loss_info_cnt += 1

        # 对每个epoch的loss取均值
        if(loss_info_cnt > 0):
            total_loss_list.append(loss / loss_info_cnt)
            xy_loss_list.append(xy_loss / loss_info_cnt)
            wh_loss_list.append(wh_loss / loss_info_cnt)
            conf_loss_list.append(conf_loss / loss_info_cnt)
        #  =============================== get loss info ==================================

    # 存储为 shape == (epoch, 4)
    no_defect_img_acc = np.reshape(no_defect_img_acc, (-1, thres_num))
    defect_img_acc = np.reshape(defect_img_acc, (-1, thres_num))
    bbox_acc = np.reshape(bbox_acc, (-1, thres_num))
    bbox_recall = np.reshape(bbox_recall, (-1, thres_num))
    # 画布
    plt.figure(figsize=[20, 20])
    # 画 thres_num 个acc图
    for thres_idx, thres in enumerate(thres_list):
        ax = plt.subplot(figureRow, figureCol, 1 + thres_idx)
        ax.set(xlim=[args.start_epoch, args.end_epoch], ylim=[0, 1], title='img/bbox accuracy of thres{}'.format(thres), ylabel='acc',xlabel='epoch')

        x = np.arange(args.start_epoch, args.end_epoch, args.ep_sample_iter)
        # x = np.linspace(args.start_epoch, args.end_epoch, num = args.end_epoch // args.ep_sample_iter)
        if args.show_no_defect_image_acc == 'True':
            plt.plot(x, no_defect_img_acc[:, thres_idx], color='blue', linewidth=1, label='no_defect_img_acc')
        if args.show_defect_image_acc == 'True':
            plt.plot(x, defect_img_acc[:, thres_idx], color='yellow', linewidth=1, label='defect_img_acc')
        if args.show_bbox_acc == 'True':
            plt.plot(x, bbox_acc[:, thres_idx], color='black', linewidth=1, label='bbox_acc')
        if args.show_bbox_recall == 'True':
            plt.plot(x, bbox_recall[:, thres_idx], color='cyan', linewidth=1, label='bbox_recall')

        plt.rcParams['font.size'] = 8
        ax.legend(fontsize=2)
        plt.legend()

    if args.show_loss == 'True':
        # 画一个loss图
        ax = plt.subplot(figureRow, figureCol, figureRow * figureCol) # loss信息放在画布的最后一个
        ax.set(xlim=[args.start_epoch, args.end_epoch], ylim=[0, 2], title='loss', ylabel='loss', xlabel='epoch')
        x = np.arange(args.start_epoch, args.end_epoch)
        plt.plot(x, total_loss_list[args.start_epoch:args.end_epoch], color='blue', linewidth=1, label='total_loss')
        plt.plot(x, xy_loss_list[args.start_epoch:args.end_epoch], color='yellow', linewidth=1, label='xy_loss')
        plt.plot(x, wh_loss_list[args.start_epoch:args.end_epoch], color='black', linewidth=1, label='wh_loss')
        plt.plot(x, conf_loss_list[args.start_epoch:args.end_epoch], color='cyan', linewidth=1, label='conf_loss')
        plt.rcParams['font.size'] = 8
        ax.legend(fontsize=2)
        plt.legend()

    # Done
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
    # plt.savefig(osp.join(args.save_path, 'analysis_log.png'))
    plt.savefig('test.png')
    print('Done')

if __name__ == '__main__':
    # print(sys.argv)
    main(sys.argv[1:])