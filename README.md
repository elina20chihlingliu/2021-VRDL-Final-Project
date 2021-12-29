# 2021-VRDL-Final-Project
The Nature Conservancy Fisheries Monitoring

The challenge is [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data) on kaggle
Final project which contains four parts:

1. Download the released annotation file of Nature Conservancy Fisheries Monitoring dataset for the bounding box of objects in the train images
2. The classes in the provided data are 'ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK','YFT'. We removed the 'OTHER' class in order to train the model with more precision.
3. Classify the digits of bounding boxes into 6 classes (0-5)

The giving Nature Conservancy Fisheries Monitoring dataset contains 3013 images for training, 1000 images in test_stg1, 12153 images in test_stg2. This project uses the YOLOv5 pre-trained model to fix this challenge.

### File descriptions
train.zip - zipped folder of all train images. The train folders are organized by fish species labels.
test_stg1.zip - zipped folder of all test images in stage 1
test_stg2.zip - zipped folder of all test images in stage 2 (not available until the second stage of the competition)
sample_submission_stg1.csv - a sample submission file in the correct format


### Environment
- Microsoft win10
- Python 3.7
- Pytorch 1.10.0
- CUDA 10.2

### YOLOv5
The project is implemented based on yolov5.
- [YOLOv5](https://github.com/ultralytics/yolov5)

## All steps including data preparation, train phase and detect phase
1. [Installation](#install-packages)
2. [Data Preparation](#data-preparation)
3. [Set Configuration](#set-configuration)
4. [Download Pretrained Model](#download-pretrained-model)
5. [Training](#training)
6. [Testing](#testing)
7. [Reference](#reference)

### Install Packages
- install pytorch from https://pytorch.org/get-started/locally/
- install openCV
```
pip install python-opencv
```
- install dependencies
```
pip install -r requirements.txt
```

### Data Preparation
Download the given dataset from [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data).

The files in the data folder is reorganized as below:
```
./data
 ├── images
 │     ├── img_00001.png
 │     │   ...
 │     └── img_03013.png
 │ 
 ├── test
 │     ├──  test_stg1
 │     │      ├── img_00001.png
 │     │      │   ...
 │     │      └── img_01000.png
 │     │
 │     └──  test_stg2
 │            ├── img_00001.png
 │            │   ...
 │            └── img_12153.png
 │   
 ├── mat_to_yolo.py
 ├── train_val_test.py
 └── coco.yaml
```


And run command `python train_val_test.py` to create train.txt, val.txt, test.txt for training and reorganize the  data structure as below:
```
./data
 ├── images
 │     ├── img_00001.png
 │     │   ...
 │     └── img_03013.png
 │
 ├── test
 │     ├──  test_stg1
 │     │      ├── img_00001.png
 │     │      │   ...
 │     │      └── img_01000.png
 │     │
 │     └──  test_stg2
 │            ├── img_00001.png
 │            │   ...
 │            └── img_12153.png
 ├── ImageSets
 │     ├──  train.txt
 │     ├──  test.txt
 │     └──  val.txt
 ├── train.txt
 ├── test.txt
 ├── val.txt
 ├── mat_to_yolo.py
 ├── train_val_test.py
 └── coco.yaml
```


And run command `python mat_to_yolo.py` to transform the .json format into yolo format in a .txt file. The transform formula for bounding boxes is: 

```
x = (xmin + (xmax - xmin)/2) * 1/image_w
y = (ymin + (ymax - ymin)/2) * 1/image_h
w = (xmax - xmin) * 1/image_w
h = (ymax - ymin) * 1/image_h
```

Reorganize the train data labels structure as below:
```
- data/labels/
├── img_00001.txt
├── img_00002.txt
│     .
│     .
│     .
└── img_03013.txt
```
### Set Configuration
- change `coco.yaml` in `./data`
```
# train, val and test data: 
train: train.txt  
val: val.txt  
test: test.txt  

# number of classes
nc: 6  

# class names
names: ['ALB','BET','DOL','LAG','SHARK','YFT'] 
```

### Download Pretrained Model
- yolov5m.pt： https://github.com/ultralytics/yolov5/releases

### Training
- train model with pretrained model
```
python train.py --epochs 10 --weights yolov5m.pt
```
### Testing
- detect test data
```
python detect.py --source data/test/test_stg1 --weights runs/train/exp/weights/best.pt
```
```
python detect.py --source data/test/test_stg2 --weights runs/train/exp/weights/best.pt
```

- Make Submission: create new_submission.csv combining test_stg1 and test_stg2 results. 
  '\n'Open yolo2coco.py, and change the test1_root_path and test2_root_path to the labels path
```
test1_root_path = 'yolov5-master/runs/detect/exp/labels'
test2_root_path = 'yolov5-master/runs/detect/exp20/labels'
```
- Transform the yolo format into coco format and combine test_stg1 & test_stg2 results:
```
python yolo2coco.py
```


### Reference
#### Related Work
- [Deep Learning for Practical Image Recognition](https://www.researchgate.net/publication/326503174_Deep_Learning_for_Practical_Image_Recognition_Case_Study_on_Kaggle_Competitions)
- [The Nature Conservancy Fisheries Monitoring Competition, 1st Place Winner’s Interview](https://medium.com/kaggle-blog/the-nature-conservancy-fisheries-monitoring-competition-1st-place-winners-interview-team-79aefc688fb)
- [CNN models with advantages and disadvantages](https://tejasmohanayyar.medium.com/a-practical-experiment-for-comparing-lenet-alexnet-vgg-and-resnet-models-with-their-advantages-d932fb7c7d17)

#### Proposed Approach
- [YOLOv5 annotation dataset](https://github.com/autoliuweijie/Kaggle/tree/master/NCFM/datasets?fbclid=IwAR2UegoKzjlkndkndXkKKDqNeqi3e4EGUWy19jya6RRGpPzLoK8s5ZxI_W8)
- [YOLOv5 Model Architecture](https://www.researchgate.net/figure/The-network-architecture-of-Yolov5-It-consists-of-three-parts-1-Backbone-CSPDarknet_fig1_349299852)
- [YOLOv5 Loss Function](https://bbs.cvmart.net/articles/3686)
- [YOLOv5 data clipping](https://www.kaggle.com/sbrugman/tricks-for-the-kaggle-leaderboard)
- [CNN Model Architecture](https://www.kaggle.com/zfturbo/fishy-keras-lb-1-25267/script)

