# -*- coding: utf-8 -*-
"""yolo2coco.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WIbQju6v9PNU9VEeU3U5Fu5XnY_yZ4AR
"""

import os
import cv2
import csv
from tqdm import tqdm
import argparse
import pandas as pd

test1_root_path = 'yolov5-master/runs/detect/exp_epoch20/labels'
test2_root_path = 'yolov5-master/runs/detect/exp18_epoch20/labels'

#test1
print("Loading data from ",test1_root_path)

img_path = 'yolov5-master/data/test/test_stg1'
assert os.path.exists(test1_root_path)
originLabelsDir = os.path.join(test1_root_path)                                        
originImagesDir = os.path.join(img_path)

# label dir name	
indexes = os.listdir(originImagesDir)

# write a row to the csv file
ptable = pd.DataFrame(columns=['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK','YFT'])

# 标注的id
for k, index in enumerate(tqdm(indexes)):
    ptable.loc[k, 'image'] = index
    ptable.loc[k, 'NoF'] = 0.123081 # guess the answer to Nof
    ptable.loc[k, 'OTHER'] = 0.079142 # guess the answer to OTHER
    result = [0,0,0,0,0,0,0,0]

    # 支持 png jpg 格式的图片。
    txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
    index = index.split(".")[0]
    # 读取图像的宽和高
    im = cv2.imread(os.path.join(img_path, (index +'.jpg')))
    height, width, _ = im.shape

    if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
        # 如没标签 
        ptable.loc[k, 'ALB'] = result[0]
        ptable.loc[k, 'BET'] = result[1]
        ptable.loc[k, 'DOL'] = result[2]
        ptable.loc[k, 'LAG'] = result[3]
        ptable.loc[k, 'SHARK'] = result[4]
        ptable.loc[k, 'YFT'] = result[5]
        continue

    with open(os.path.join(originLabelsDir, (index +'.txt')), 'r') as fr:
        labelList = fr.readlines()
        for label in labelList:
            dataset = {}
            label = label.strip().split()
            cls_id = int(label[0])
            score = float(label[5])

            if result[cls_id] < score:
                result[cls_id] = score

        ptable.loc[k, 'ALB'] = result[0]
        ptable.loc[k, 'BET'] = result[1]
        ptable.loc[k, 'DOL'] = result[2]
        ptable.loc[k, 'LAG'] = result[3]
        ptable.loc[k, 'SHARK'] = result[4]
        ptable.loc[k, 'YFT'] = result[5]

#test2
print("Loading data from ",test2_root_path)

img_path = 'yolov5-master/data/test/test_stg2'
assert os.path.exists(test2_root_path)
originLabelsDir = os.path.join(test2_root_path)                                        
originImagesDir = os.path.join(img_path)

# label dir name
indexes = os.listdir(originImagesDir)
indexes = os.listdir(originImagesDir)

# 标注的id
for k, index in enumerate(tqdm(indexes),start = 1000):
    ptable.loc[k, 'image'] = "test_stg2/" + index
    ptable.loc[k, 'NoF'] = 0.123081 # guess the answer to Nof
    ptable.loc[k, 'OTHER'] = 0.079142 # guess the answer to OTHER
    result = [0,0,0,0,0,0,0,0]

    # 支持 png jpg 格式的图片。
    txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')
    index = index.split(".")[0]
    # 读取图像的宽和高
    im = cv2.imread(os.path.join(img_path, (index +'.jpg')))
    height, width, _ = im.shape

    if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
        # 如没标签 
        ptable.loc[k, 'ALB'] = result[0]
        ptable.loc[k, 'BET'] = result[1]
        ptable.loc[k, 'DOL'] = result[2]
        ptable.loc[k, 'LAG'] = result[3]
        ptable.loc[k, 'SHARK'] = result[4]
        ptable.loc[k, 'YFT'] = result[5]
        continue

    with open(os.path.join(originLabelsDir, (index +'.txt')), 'r') as fr:
        labelList = fr.readlines()
        for label in labelList:
            dataset = {}
            label = label.strip().split()
            cls_id = int(label[0])
            score = float(label[5])

            if result[cls_id] < score:
                result[cls_id] = score
                
        ptable.loc[k, 'ALB'] = result[0]
        ptable.loc[k, 'BET'] = result[1]
        ptable.loc[k, 'DOL'] = result[2]
        ptable.loc[k, 'LAG'] = result[3]
        ptable.loc[k, 'SHARK'] = result[4]
        ptable.loc[k, 'YFT'] = result[5]

# 保存结果
save_path = 'yolov5-master/runs/detect/final_submission/submission.csv'
ptable.to_csv(save_path, index=False)
print('Save annotation to {}'.format(save_path))

import pandas as pd
import os

clip = 0.85
classes = 8
save_path = 'yolov5-master/runs/detect/final_submission/submission.csv'

def clip_csv(csv_file, clip, classes):

    df_image = pd.read_csv(csv_file , usecols=["image"])

    # Read the submission file
    df = pd.read_csv(csv_file , usecols=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK','YFT'])
    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)
    
    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    #merge two dataframe
    df_final = df_image.merge(df , how = 'inner' , left_index= True, right_index = True)

    return df_final
    
# Of course you are going to use your own submission here
new_submission = clip_csv(save_path, clip, classes)

save_path = 'yolov5-master/runs/detect/final_submission/new_submission.csv'
new_submission.to_csv(save_path, index=False)
print('Save annotation to {}'.format(save_path))