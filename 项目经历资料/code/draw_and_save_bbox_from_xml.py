import os
from os import path
import xml.dom.minidom
import cv2 as cv
import pdb
from tqdm import tqdm
import argparse
import pdb
import numpy as np


"""
root
│
└───sub_folder1
│   │   file011.jpg
│   │   file012.jpg
│   │   ...
│   │   file013.jpg
│   │   xml
│       │*.xml
│       │*.xml
│
└───sub_folder2
│   │   file011.jpg
│   │   file012.jpg
│   │   ...
│   │   file013.jpg
│   │   xml
│       │*.xml
│       │*.xml

"""

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", help="path to folder containing src_images",default='/home-ex/tclhk/chenww/datasets/t7/labeled_total_v3/')
parser.add_argument("--tgt_dir", help="path to folder containing tgt_images",default='/home-ex/tclhk/chenww/datasets/t7/labeled_total_v3_crop/')
param = parser.parse_args()



def process_to_sqr(img,x1,y1,x2,y2):
    img_rows=img.shape[0]
    img_cols=img.shape[1]

    length = x2 - x1
    width = y2 - y1
    center = [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 2)]

    extend = round(max(length, width) / 2 + max(length, width) * 0.15)
    new_x1, new_x2, new_y1, new_y2 = center[0] - extend, center[0] + extend, center[1] - extend, center[1] + extend  # try use lambda ,map

    new_x1 = 0 if new_x1 < 0 else new_x1
    new_y1 = 0 if new_y1 < 0 else new_y1
    new_x2 = img_cols if new_x2 > img_cols else new_x2
    new_y2 = img_rows if new_y2 > img_rows else new_y2

    return new_x1, new_y1, new_x2, new_y2

def process_to_224_from_xyxy(img,x1,y1,x2,y2):
    img_rows=img.shape[0]
    img_cols=img.shape[1]

    h = img.shape[1]
    w = img.shape[0]
    # center = [x1 + round((x2 - x1) / 2), y1 + round((y2 - y1) / 2)]
    cx = x1 + round((x2 - x1) / 2)
    cy = y1 + round((y2 - y1) / 2)


    sz = np.random.randint(-30, 30) + 224
    new_x1 = np.random.randint(-80, 80) + cx - sz//2
    new_y1 = np.random.randint(-80, 80) + cy - sz//2

    new_x1 = min(max(0, new_x1), 2048 - sz)
    new_y1 = min(max(0, new_y1), 2048 - sz)

    return new_x1, new_y1, new_x1 + sz, new_y1 + sz


# 输入批量图片的路径，及批量XML文件的路径，保存画出bounding box的图片及保存裁减出bounding box的图片
# imgPath : img/*.jpg, *.jpg, *.jpg ......
# annoPath : xml/*.xml, *.xml, *.xml ......
# img_fmt : 图片的格式
# TODO:借鉴谢舰的代码优化
def draw_and_save_bbox_from_xml(imgPath, annoPath, img_fmt, draw_bbox_saved_path, bbox_saved_path):
    imagelist = os.listdir(imgPath)
    # f = open('./edge_err2.txt','a')
    for image in imagelist:
        if os.path.splitext(image)[-1] == img_fmt:
            # pdb.set_trace()
            image_pre, ext = os.path.splitext(image) # 图片名称前缀及后缀
            imgfile = imgPath +'/'+ image
            xmlfile = annoPath + image_pre + '.xml'
            # print(image)
            # 打开xml文档
            DOMTree = xml.dom.minidom.parse(xmlfile)
            # 得到文档元素对象
            collection = DOMTree.documentElement
            # 读取图片
            # print(imgfile)
            img = cv.imread(imgfile)
            h = img.shape[0]
            w = img.shape[1]
            print('h', h)
            print('w', w)

            filenamelist = collection.getElementsByTagName("filename")
            filename = filenamelist[0].childNodes[0].data
            # print(filename)
            # 得到标签名为object的信息
            objectlist = collection.getElementsByTagName("object")

            for i, objects in enumerate(objectlist):
                # 每个object中得到子标签名为name的信息
                namelist = objects.getElementsByTagName('name')
                # 通过此语句得到具体的某个name的值
                objectname = namelist[0].childNodes[0].data

                bndbox = objects.getElementsByTagName('bndbox')
                if not bndbox:
                    # print(image)
                    pass
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                    x2 = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(y2_list[0].childNodes[0].data)

                    new_x1, new_x2, new_y1, new_y2 = process_to_224_from_xyxy(img, x1, y1, x2, y2)
                    print([new_x1, new_x2, new_y1, new_y2])

                    if len(objectlist)>1: # more than one bouding box
                        img_1 = img.copy()
                        print(img_1)

                        bbox = img_1[new_y1:new_y2,new_x1:new_x2]
                        print(img_1[new_y1:new_y2,new_x1:new_x2])
                        cv.resize(img_1[new_y1:new_y2,new_x1:new_x2], (224, 224))
                        cv.imwrite(bbox_saved_path + '/' +image_pre+'_'+str(i)+ext, bbox)

                        cv.rectangle(img_1, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), thickness=4)
                        cv.imwrite(draw_bbox_saved_path + '/' +image_pre+str(i)+ext,img_1)
                    else:
                        # print(img)
                        # pdb.set_trace()
                        bbox = img[new_y1:new_y2,new_x1:new_x2]
                        print(img[new_y1:new_y2,new_x1:new_x2])
                        # cv.resize(img[new_y1:new_y2,new_x1:new_x2], (224, 224))
                        cv.imwrite(bbox_saved_path + '/' +filename, bbox)

                        # cv.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), thickness=4)
                        # cv.imwrite(draw_bbox_saved_path + '/' +filename, img)
                        print(image)



def single_draw_anchor(imgFile,annoFile,save_path):
    # print(image)
    # 打开xml文档
    DOMTree = xml.dom.minidom.parse(annoFile)
    # 得到文档元素对象
    collection = DOMTree.documentElement
    # 读取图片
    img = cv.imread(imgFile)
    img_square = img
    img_15 = img

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    filenamelist = collection.getElementsByTagName("filename")
    cls = collection.getElementsByTagName("folder")
    print(cls)
    filename = filenamelist[0].childNodes[0].data
    # print(filename)
    # 得到标签名为object的信息
    objectlist = collection.getElementsByTagName("object")

    for objects in objectlist:
        # 每个object中得到子标签名为name的信息
        namelist = objects.getElementsByTagName('name')
        # 通过此语句得到具体的某个name的值
        objectname = namelist[0].childNodes[0].data

        bndbox = objects.getElementsByTagName('bndbox')
        # print(bndbox)
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)

            length = x2 - x1
            width = y2 - y1
            center = [x1 + round((x2-x1)/2), y1 + round((y2-y1)/2)]
            extend = round(max(length, width) / 2 + max(length, width) * 0.15)
            new_x1, new_x2, new_y1, new_y2 = center[0] - extend, center[0] + extend, center[1] - extend, center[1] + extend # try use lambda ,map

            if new_x1 < 0 or new_y1 < 0 or new_x2 > img_cols or new_y2 > img_rows:
                print('ERROR:',filename)
            else:
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                cv.imwrite(save_path + filename, img)  # save picture

                cv.rectangle(img_15, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), thickness=2)
                # cv.imwrite(save_path + 'square15.jpg', img_15)  # save picture

def main():
    for root, sub_folders, files in os.walk(param.src_dir):
        for sub_folder in sub_folders:  # 遍历所有子文件夹,各个类别
            if sub_folder != 'xml' or sub_folder != 'yolo_txt' or sub_folder != 'txt_xyhw':
                new_process_cls_path = os.path.join(param.tgt_dir, sub_folder)
                draw_bbox_saved_path = os.path.join(param.tgt_dir, sub_folder + '_draw_bbox')
                bbox_saved_path = os.path.join(param.tgt_dir, sub_folder)

            if not os.path.exists(new_process_cls_path):
                os.makedirs(new_process_cls_path)
            if not os.path.exists(draw_bbox_saved_path):
                os.makedirs(draw_bbox_saved_path)
            if not os.path.exists(bbox_saved_path):
                os.makedirs(bbox_saved_path)

            imgPath = os.path.join(root, sub_folder)
            draw_and_save_bbox_from_xml(imgPath, os.path.join(root, sub_folder + '/xml/'), ".jpg", draw_bbox_saved_path, bbox_saved_path)




if __name__ == '__main__':
    main()