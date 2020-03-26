import sys
import random
import os.path as osp

def get_txt_data(txt_file, cls_list):
    with open(txt_file) as f:
        lines = f.readlines()

    if cls_list == []:
        return lines
    else:
        c_lines = []
        for line in lines:
            dat_list = line.strip().split()
            if dat_list[1] in cls_list:
                c_lines.append(line)
        return c_lines

def main():
    dat_path = '/home-ex/tclhk/chenww/t2/yolo_v3_x/config/d10/temp/train.dat'
    output_dat_path = '/home-ex/tclhk/chenww/t2/yolo_v3_x/config/d10/temp/'
    if not osp.exists(output_dat_path):
        os.makedirs(output_dat_path)

    cmd_num = int(sys.argv[1])
    cmd_cls_list = sys.argv[2:]

    lines = get_txt_data(dat_path, cmd_cls_list)
    random.shuffle(lines)

    assert len(lines) > cmd_num

    idx_list = random.sample(range(1,len(lines)), cmd_num)

    output_dat_path = output_dat_path + 'train{}'.format(cmd_num) + '.dat'
    with open(output_dat_path, 'w') as f:
        for i in range(len(idx_list)):
            f.write(lines[i])

    print("done")


if __name__ == '__main__':
    main()


