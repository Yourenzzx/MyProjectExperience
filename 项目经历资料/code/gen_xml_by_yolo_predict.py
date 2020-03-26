import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
import json
import numpy as np
import torch
import shutil
import argparse
import sys
import os.path as osp

from torch.autograd import Variable
from tqdm import tqdm
from yolo_v3_x.model import ResNet as det_resnet
from yolo_v3_x.utils import non_max_suppression
from classification_x.model import ResNet as cla_resnet
from xml_lib.AbstractXMLWriterHandler import AbstractXMLWriterHandler
from xml_lib.XMLTag import XMLTag
from xml_lib.XMLWriter import XMLWriter



class Predict(object):
    def __init__(self, config):
        self.config = config
        self.detect_config = config['detect_config']
        self.classify_config = config['classify_config']
        self.eval_vis_path = config['eval_vis_path']
        self.det_model = self.get_det_model(self.detect_config)
        # self.classify_model, self.class_name = self.get_classify_model(self.classify_config)
        # self.class_name.append('TSFAS')
        self.detect_result_dict = {}
        # 形式为{image_path:[{bndbox:box,det_conf:score}]},box为解归一化的左上右下
        self.final_resutl_dict = {}
        # 形式为{image_path:{label:name,bndbox:box,conf:sore}}TSFAS类bodbox为[],正常为解归一化的左上右下

    def get_det_model(self, config):
        model_dict = torch.load(config['model_weight'])
        anchors = model_dict['anchors'].to('cuda')
        model = det_resnet(anchors, Istrain=False).to('cuda')
        model.load_state_dict(model_dict['net'])
        model.eval()
        return model

    def inference_det_model(self, img):
        process_img = cv2.resize(img, self.detect_config['process_size'])
        inputs = process_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        inputs = np.expand_dims(inputs, 0)
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs, requires_grad=False).to('cuda')
        with torch.no_grad():
            _, outputs = self.det_model(inputs)
            outputs = non_max_suppression(outputs, conf_thres=self.detect_config['conf_thres'],
                                          nms_thres=self.detect_config['nms_thres'])
            outputs_numpy = []
            for output in outputs:
                if output is None:
                    outputs_numpy.append(None)
                else:
                    outputs_numpy.append(output.detach().cpu().numpy())
        assert len(outputs_numpy) == 1
        return outputs_numpy


    def draw_box_in_img(self, image_path, box_dict):
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        if len(box_dict['bndbox']) > 0:
            x1, y1, x2, y2 = box_dict['bndbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            xx = min(x1, w - 80)
            yy = y1 - 5 if y1 > 30 else y2 + 25
            if y1 < 30 and y2 > h - 25: yy = h // 2
            cv2.putText(img, '{:.2f}'.format(box_dict['conf']), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, \
                        0.85, (0, 0, 255), 1)
        return img

def build_xml(file_name, width, height, xmin, ymin, xmax, ymax, output_xml_path):
    root = XMLTag(name="annotation", parent=None)

    root.addChild(XMLTag("filename", None, file_name, parent=root))

    root.addChild(XMLTag("size", parent=root))
    root.children[1].addChild(XMLTag("width", None, str(width), root.children[0]))
    root.children[1].addChild(XMLTag("height", None, str(height), root.children[0]))
    root.children[1].addChild(XMLTag("depth", None, '3', root.children[0]))

    root.addChild(XMLTag("object", None, None, root))
    root.children[2].addChild(XMLTag("name", None, 'bbox', root.children[1]))
    root.children[2].addChild(XMLTag("bndbox", None, None, root.children[1]))

    root.children[2].children[1].addChild(XMLTag("xmin", None, str(xmin), root.children[1].children[1]))
    root.children[2].children[1].addChild(XMLTag("ymin", None, str(ymin), root.children[1].children[1]))
    root.children[2].children[1].addChild(XMLTag("xmax", None, str(xmax), root.children[1].children[1]))
    root.children[2].children[1].addChild(XMLTag("ymax", None, str(ymax), root.children[1].children[1]))

    # Constructing to pointers.xml
    handler = AbstractXMLWriterHandler(root)
    writer = XMLWriter(output_xml_path, handler)
    writer.write()


def is_valid_image(path):
  '''
  检查文件是否损坏
  '''
  try:
    bValid = True
    fileObj = open(path, 'rb') # 以二进制形式打开
    buf = fileObj.read()
    if not buf.startswith(b'\xff\xd8'): # 是否以\xff\xd8开头
      bValid = False
    elif buf[6:10] in (b'JFIF', b'Exif'): # “JFIF”的ASCII码
      if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'): # 是否以\xff\xd9结尾
        bValid = False
    else:
      try:
        Image.open(fileObj).verify()
      except Exception as e:
        bValid = False
        print(e)
  except Exception as e:
    return False
  return bValid

def gen_xml_by_yolo_predict(predict, file_path, xml_save_path):
    # if img is None: return
    try:
        img = cv2.imread(file_path)
        if img is None: return
        h, w, _ = img.shape
    except:
        print(file_path)


    det_single_result = predict.inference_det_model(img)[0]
    predict.detect_result_dict[file_path] = []
    if det_single_result is None: # TSFAS
        inference_box = []
        pre_class_name = 'TSFAS'
        score = 0.3
    else: # OTS
        box_index = np.argmax(det_single_result[:, 4] * (det_single_result[:, 2] + det_single_result[:, 3] - det_single_result[:, 0] - det_single_result[:, 1]))
            # predict.detect_result_dict[file_path].append({'bndbox': box[0:4], 'conf': float(box[4])})

        x1_, y1_, x2_, y2_, _ = det_single_result[box_index]
        cx = (x1_ + x2_) / 2
        cy = (y1_ + y2_) / 2

        # ============= 原始框 =============
        x1 = int(x1_ * w )
        y1 = int(y1_ * h )
        x2 = int(x2_ * w)
        y2 = int(y2_ * h)
        # ============= 原始框 =============

        # ============= 224*224 =============
        # crop_size = predict.classify_config['crop_size']
        # x1 = int(cx * w - crop_size / 2)
        # y1 = int(cy * h - crop_size / 2)
        # x1 = min(max(0, x1), w - crop_size)
        # y1 = min(max(0, y1), h - crop_size)
        # x2 = x1 + crop_size
        # y2 = y1 + crop_size
        # ============= 224*224 =============

        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.imwrite('./2.jpg', img)

        img_file_name = osp.basename(file_path)
        # xml保存在图片所在的文件夹
        # xml_save_path = osp.join(file_path.split(img_file_name)[0], img_file_name.split('.jpg')[0]+'.xml')
        # xml另存为单独的文件夹
        # xml_save_path = osp.join('/home-ex/tclhk/chenww/datasets/t2/online/0226/4days/xml/', file_path.split('/')[10])
        # if not osp.exists(xml_save_path):
        #     os.makedirs(xml_save_path)
        # assert not osp.exists(xml_save_path)

        build_xml(img_file_name, w, h, x1, y1, x2, y2, osp.join(xml_save_path, img_file_name.split('.jpg')[0] + '.xml'))

def isInsidePanel(img_name):
    arr = img_name.split('_')
    x = float(arr[4])
    y = float(arr[5])

    ver_bd = ['-1224.1', '-14.2', '14.2', '1224.1']
    hor_bd = ['-1063.2', '-383.1', '-339.9', '340.2', '383.4', '1063.5']

    cond1 = (y > float(ver_bd[0]) and y < float(ver_bd[1]) and x > float(hor_bd[0]) and x < float(hor_bd[1]))
    cond2 = (y > float(ver_bd[0]) and y < float(ver_bd[1]) and x > float(hor_bd[2]) and x < float(hor_bd[3]))
    cond3 = (y > float(ver_bd[0]) and y < float(ver_bd[1]) and x > float(hor_bd[4]) and x < float(hor_bd[5]))
    cond4 = (y > float(ver_bd[2]) and y < float(ver_bd[3]) and x > float(hor_bd[0]) and x < float(hor_bd[1]))
    cond5 = (y > float(ver_bd[2]) and y < float(ver_bd[3]) and x > float(hor_bd[2]) and x < float(hor_bd[3]))
    cond6 = (y > float(ver_bd[2]) and y < float(ver_bd[3]) and x > float(hor_bd[4]) and x < float(hor_bd[5]))
    if (cond1 or cond2 or cond3 or cond4 or cond5 or cond6):
        return True
    else:
        return False

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_weight", type=str, default="/home-ex/tclhk/chenww/t2/models/yolo_v3_x/0211_v7_2/yolov3_ckpt_21.pth")
    parser.add_argument("--det_conf_th", type=float, default=0.3)
    parser.add_argument("--det_nms_th", type=float, default=0.5)

    parser.add_argument("--cls_model_weight", type=str)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--eval_img_path", type=str, default="/home-ex/tclhk/chenww/datasets/t2/d10/2253-D10-0112~0212/insidePanel/")
    parser.add_argument("--xml_save_root", type=str, default="/home-ex/tclhk/chenww/datasets/t2/d10/2253-D10-0112~0212/xml/")
    parser.add_argument("--eval_gt_path", type=str)
    parser.add_argument("--eval_vis_path", type=str)

    args = parser.parse_args(argv)

    detect_config =   {'model_weight': args.det_model_weight,
                       'conf_thres': args.det_conf_th, 'nms_thres': args.det_nms_th, 'process_size': (1024, 768)}
    classify_config = {'model_weight': args.cls_model_weight,
                       'crop_size': args.crop_size}
    config_dict =     {'classify_config': classify_config,
                       'detect_config': detect_config,
                       'eval_img_path': args.eval_img_path,
                       'xml_save_root': args.xml_save_root,
                       'eval_gt_path': args.eval_gt_path,
                       'eval_vis_path': args.eval_vis_path}

    predict = Predict(config_dict)

    eval_img_path = predict.config['eval_img_path']
    xml_save_root = predict.config['xml_save_root']
    gt_clses = os.listdir(eval_img_path)

    # for gt_cls in tqdm(gt_clses):
    #     pred_clses = os.listdir(osp.join(eval_img_path, gt_cls))
    #     for pred_cls in pred_clses:
    #         if pred_cls == 'OUT':
    #             shutil.rmtree(osp.join(eval_img_path, gt_cls, pred_cls))
    #             continue
    #         imgs_list = os.listdir(osp.join(eval_img_path, gt_cls, pred_cls))
    #         for img in imgs_list:

                # copy file to a folder
                # dir = '/home-ex/tclhk/chenww/datasets/t2/online/2250-D13-20200114.txt_OUT_for_train_classification/'
                # if not osp.exists(dir):
                #     os.makedirs(dir)
                # img_path = osp.join(eval_img_path, gt_cls, pred_cls, img)
                # img_save_as_dir = osp.join(dir, gt_cls)
                # if not osp.exists(img_save_as_dir):
                #     os.makedirs(img_save_as_dir)
                # shutil.copyfile(img_path, osp.join(img_save_as_dir, img))


                # if img.endswith('.xml'): continue
                # img_path = osp.join(eval_img_path, gt_cls, pred_cls, img)
                # # gen_xml_by_yolo_predict(predict, img_path)
                # if not is_valid_image(img_path):
                #     print("unvalid image:", img_path)
                #     os.remove(img_path)
                # else:
                #     gen_xml_by_yolo_predict(predict, img_path)

    for gt_cls in tqdm(gt_clses):
        imgs_list = os.listdir(osp.join(eval_img_path, gt_cls))
        if gt_cls == 'TSFAS':
            continue
        for img in imgs_list:
            if img.endswith('.xml') or img.endswith('.db'): continue
            img_path = osp.join(eval_img_path, gt_cls, img)
            if not is_valid_image(img_path):
                print("unvalid image:", img_path)
                os.remove(img_path)
            else:
                # if isInsidePanel(img):
                xml_save_path = osp.join(xml_save_root, gt_cls)
                if not osp.exists(xml_save_path): os.makedirs(xml_save_path)
                gen_xml_by_yolo_predict(predict, img_path, xml_save_path)




if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])