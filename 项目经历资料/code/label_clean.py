import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print('pid:{}   GPU:{}'.format(os.getpid(), os.environ['CUDA_VISIBLE_DEVICES']))
import cv2
import json
from yolov3.model import ResNet as det_resnet
from yolov3.utils import non_max_suppression
from classification.model import ResNet as cla_resnet
import numpy as np
np.set_printoptions(suppress=True)
import torch
from torch.autograd import Variable
import shutil
from tqdm import tqdm

test_image_path = '/home-ex/tclhk/xie/Data_1350/dataset_new/train/TPDPD/'


# 联合测试，由检测模型和分类模型给出联合结果
detect_config = {'model_weight': '/home-ex/tclhk/xie/ADC-code/yolov3/models/weights_1350_0108/yolov3_ckpt_33.pth',
                 'conf_thres': 0.3, 'nms_thres': 0.5, 'process_size': (1024, 768)}
classify_config = {
    'model_weight': '/home-ex/tclhk/xie/ADC-code/classification/models/weights_1350_0108/model_ckpt_163.pth',
    'crop_size': 224, }

config_dict = {'classify_config': classify_config,
               'detect_config': detect_config,
               'eval_img_path': '/home-ex/tclhk/xie/ADC-code/yolov3/config/data-0108/val.dat',
               # 'eval_img_path': '/home-ex/tclhk/xie/Data_225x/225x/dataset_1/val/',
               'result_save_path': '/home-ex/tclhk/xie/ADC-code/dataset/',
               # 'eval_gt_path': None,
               'eval_gt_path': '/home-ex/tclhk/xie/ADC-code/yolov3/config/data-0108/val.dat',
               'eval_vis_path': '/home-ex/tclhk/xie/ADC-code/dataset/confusion/'}


class Predict(object):
    def __init__(self, config):
        self.config = config
        self.detect_config = config['detect_config']
        self.classify_config = config['classify_config']
        self.eval_vis_path = config['eval_vis_path']
        self.det_model = self.get_det_model(self.detect_config)
        self.classify_model, self.class_name = self.get_classify_model(self.classify_config)
        self.class_name.append('TSFAS')
        self.detect_result_dict = {}
        # 形式为{image_path:[{bndbox:box,det_conf:score}]},box为解归一化的左上右下
        self.final_resutl_dict = {}
        # 形式为{image_path:{label:name,bndbox:box,conf:sore}}TSFAS类bodbox为[],正常为解归一化的左上右下

    def get_det_model(self, config):
        print('detect model weight: {}'.format(config['model_weight']))
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

    def get_classify_model(self, config):
        print('classify model weight: {}'.format(config['model_weight']))
        model_dict = torch.load(config['model_weight'])
        class_name = model_dict['class_name']

        state_dict = model_dict['net']

        model = cla_resnet(class_name=class_name)
        model.to('cuda')

        model.load_state_dict(state_dict)
        model.eval()
        return model, class_name

    def inference_cla_model(self, img, boxes):
        # img = cv2.imread(file_path)
        assert img is not None
        h, w, _ = img.shape
        # boxes_str = ' '.join([' '.join(map(str, s)) for s in boxes])
        # print(boxes_str)
        box_index = np.argmax(boxes[:, 4] * (boxes[:, 2] + boxes[:, 3] - boxes[:, 0] - boxes[:, 1]))
        x1_, y1_, x2_, y2_, _ = boxes[box_index]
        cx = (x1_ + x2_) / 2
        cy = (y1_ + y2_) / 2
        # 使用conf*宽高最大的那个作为预测结果
        crop_size = self.classify_config['crop_size']
        x1 = int(cx * w - crop_size / 2)
        y1 = int(cy * h - crop_size / 2)
        x1 = min(max(0, x1), w - crop_size)
        y1 = min(max(0, y1), h - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        img_crop = img[y1:y2, x1:x2, :]
        # cv2.imwrite('{}.jpg'.format(i), img_crop)

        inputs_numpy = np.transpose(img_crop, (2, 0, 1))
        inputs_numpy = np.expand_dims(inputs_numpy.astype(np.float32), 0)

        with torch.no_grad():
            inputs = torch.from_numpy(inputs_numpy / 255)
            inputs = Variable(inputs.to('cuda'), requires_grad=False)
            f, y = self.classify_model(inputs)
            # f为avg_pool以后的铺平特征向量
            y = torch.sigmoid(y).detach().cpu().numpy()[0]

        return y, (x1, y1, x2, y2)

    def combine_test(self, file_path):
        img = cv2.imread(file_path)
        det_single_result = self.inference_det_model(img)[0]
        if det_single_result is None:
            return None, None
        self.detect_result_dict[file_path] = []
        inference_result, inference_box = self.inference_cla_model(img, det_single_result)
        return inference_result, inference_box

if __name__ == '__main__':
    ots_error_path = '/home-ex/tclhk/xie/ADC-code/dataset/ots_confusion.txt'
    with open(ots_error_path) as f:
        data = f.readlines()
    data = [da.strip() for da in data]
    predict = Predict(config_dict)
    for dirname,folders,files in os.walk(test_image_path):
        if len(files)==0:
            continue
        for file in tqdm(files):
            if not file.endswith('.JPG'):
                continue
            # if file not in data:
            #     continue
            image_path = os.path.join(dirname,file)
            cla_res,inference_box = predict.combine_test(image_path)

            if cla_res is not None:
                res_list = []
                top_k_sort = np.argsort(cla_res)[::-1]
                for top in top_k_sort:
                    if cla_res[top]>0.1:
                        res_list.append(predict.class_name[top])
                if len(res_list)==1:
                    continue
                print(res_list)
                print([cla_res[rl] for rl in top_k_sort[:len(res_list)]])



                # if np.sum(cla_res)>1:
                # res = np.around(cla_res,6)
                # res = res[res>0.01]

                # print(cla_res)
            # assert 1==7

