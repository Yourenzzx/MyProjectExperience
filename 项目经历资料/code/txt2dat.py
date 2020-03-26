import os
import os.path as osp

img_root = '/home-ex/tclhk/chenww/datasets/t2/temp2/train/'
txt_root = '/home-ex/tclhk/chenww/datasets/t2/temp2/txt/'
output_dat_path = '/home-ex/tclhk/chenww/t2/yolo_v3_x/config/d10/temp/train2.dat'


def get_txt_data(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    data_list = [line.strip().split() for line in lines]

    return data_list

def max_box(data_list):
    max_wh = 0
    i = -1
    for idx, dat in enumerate(data_list):
        cx = float(data_list[idx][1])
        cy = float(data_list[idx][2])
        fw = float(data_list[idx][3])
        fh = float(data_list[idx][4])
        if(max_wh < fw * fh):
            max_wh = fw * fh
            i = idx

    return data_list[i][1], data_list[i][2], data_list[i][3], data_list[i][4]


gt_clses = os.listdir(img_root)
with open(output_dat_path, 'w') as f:
    for gt_cls in gt_clses:
        path = osp.join(img_root, gt_cls)
        img_list = os.listdir(path)
        for img in img_list:
            if img.endswith('.xml'): continue
            img_pre = img.split('.JPG')[0]
            txt_file = img_pre + '.txt'

            img_abs_path = osp.join(img_root, gt_cls, img)

            if gt_cls == 'TFOL0':
                f.write('{} {}\n'.format(img_abs_path, gt_cls))
                continue

            data_list = get_txt_data(osp.join(txt_root, gt_cls, txt_file))
            if data_list == []:
                print("data is empty")
                continue

            cx, cy, fw, fh = max_box(data_list)

            f.write('{} {} 1.0 {} {} {} {}\n'.format(img_abs_path, gt_cls, cx, cy, fw, fh))


