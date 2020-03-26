from __future__ import division
import os
import os.path as osp

from xml_lib.AbstractXMLWriterHandler import AbstractXMLWriterHandler
from xml_lib.XMLTag import XMLTag
from xml_lib.XMLWriter import XMLWriter




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


def get_txt_data(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    data_list = [line.strip().split() for line in lines]

    return data_list

# bbox 归一化坐标 转换成 正常坐标
# 输入： norm_box = [cx, cy, fw, fh]
def box_convert(norm_box, img_height, img_width):
    cx = norm_box[0]
    cy = norm_box[1]
    fw = norm_box[2]
    fh = norm_box[3]

    w = fw * img_width
    h = fh * img_height

    xmin = (cx - fw / 2) * img_width
    ymin = (cy - fh / 2) * img_height
    # xmax = xmin + w
    # ymax = ymin + h

    xmax = xmin + h
    ymax = ymin + w

    return int(xmin), int(ymin), int(xmax), int(ymax)


def txt2xml(img_height, img_width, txt_root, save_root):
    txt_file_list = os.listdir(txt_root)

    for txt_file in txt_file_list:
        if not txt_file.endswith('.txt'):  continue

        data_list = get_txt_data(osp.join(txt_root, txt_file))

        cx = float(data_list[0][1])
        cy = float(data_list[0][2])
        fw = float(data_list[0][3])
        fh = float(data_list[0][4])

        box_data = [cx, cy, fw, fh]
        xmin, ymin, xmax, ymax = box_convert(box_data, img_height, img_width)

        # print(xmin, ymin, xmax, ymax)

        img_pre = txt_file.split('.txt')[0]
        img_name = img_pre + '.JPG'
        xml_name = img_pre + '.xml'
        output_xml_path = osp.join(save_root, xml_name)

        build_xml(img_name, img_width, img_height, xmin, ymin, xmax, ymax, output_xml_path)


if __name__ == '__main__':
    txt_root = '/home-ex/tclhk/chenww/tools/test_data/'
    save_root = './temp2/'
    if not osp.exist(save_root):
        os.makedirs(save_root)
    txt2xml(768, 1024, txt_root, save_root)