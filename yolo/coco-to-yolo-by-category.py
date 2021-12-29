"""
reference :　https://github.com/Weifeng-Chen/DL_tools/blob/main/coco2yolo.py
COCO 格式的数据集转化为 YOLO 格式的数据集
--json_path 输入的json文件路径
--save_path 保存的文件夹名字，默认为当前目录下的labels。
"""

import os 
import json
from tqdm import tqdm
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./label/YFT.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='./labels/YFT', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    datas = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

for data in datas:
    annotations = data['annotations']
    img_path = data['filename']
    img = cv2.imread(os.path.join('.',img_path))
    img_height, img_width,_ = img.shape
    head = (os.path.splitext(img_path))[0].split("/")[1]
    ana_txt_name = head + ".txt"
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')

    for annotation in annotations:
        height = annotation['height']
        width = annotation['width']
        x = annotation['x']
        y = annotation['y']
        box = [x, y, x + width, y + height]
        
        box = convert((img_width, img_height), box)
        f_txt.write("%s %s %s %s %s\n" % (5, box[0], box[1], box[2], box[3]))
    f_txt.close()