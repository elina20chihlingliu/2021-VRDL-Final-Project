# 2021-VRDL-Final-Project
The Nature Conservancy Fisheries Monitoring
以下要修正

The proposed challenge is a street view house numbers detection, which contains two parts:
1. Do bounding box regression to find top, left, width and height of bounding boxes which contain digits in a given image
2. classify the digits of bounding boxes into 10 classes (0-9)

The giving SVHN dataset contains 33402 images for training and 13068 images for testing. This project uses the YOLOv5 pre-trained model to fix this challenge.

### Environment
- Microsoft win10
- Python 3.7
- Pytorch 1.10.0
- CUDA 10.2

### YOLOv5
The project is implemented based on yolov5.
- [YOLOv5](https://github.com/ultralytics/yolov5)

## Reproducing Submission
To reproduct my submission without retrainig, run inference.ipynb on my Google Drive:
- [inference.ipynb](https://drive.google.com/file/d/14IUxba_Tjaw3teusvljHuXGmZ8rEvH1a/view?usp=sharing)

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
Download the given dataset from [Google Drive](https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc) or [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/).

The files in the data folder is reorganized as below:
```
./data
 ├── train
 │     ├──  xxx.png
 │     └──  digitStruct.mat
 ├── test
 │     └──  yyy.png
 ├── mat_to_yolo.py
 ├── train_val_test.py
 └── shvn.yaml
```


And run command `python train_val_test.py` to create train.txt, val.txt, test.txt for training and reorganize the  data structure as below:
```
./data
 ├── train
 │     ├──  xxx.png
 │     └──  digitStruct.mat
 ├── test
 │     └──  yyy.png
 ├── dataset
 │     ├──  train.txt
 │     ├──  test.txt
 │     └──  val.txt
 ├── train.txt
 ├── test.txt
 ├── val.txt
 ├── mat_to_yolo.py
 ├── train_val_test.py
 └── shvn.yaml
```


And run command `python mat_to_yolo.py` to create labels for yolo and reorganize the train data structure as below:
```
- train/
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
│     .
│     .
│     .
├── 33402.png
└── 33402.txt
```
### Set Configuration
- create `svhn.yaml` in `./data`
```
# train, val and test data: # Total 33402 images
train: data/train.txt  # 27055 images
val: data/val.txt  # 3007 images
test: data/test.txt  # 3340 images

# number of classes
nc: 10

# class names
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
```

### Download Pretrained Model
- yolov5m.pt： https://github.com/ultralytics/yolov5/releases

### Training
- train model with pretrained model
```
python train.py --epochs 20 --weights yolov5m.pt
```
### Testing
- detect test data
```
python detect.py --source data/test/ --weights runs/train/exp/weights/best.pt
```

- Make Submission: creat answer.json following coco dataset format
```
python output.py
```
```
[{
  "image_id": image_name,
  "score": confidence,
  "category_id": predict_label,
  "bbox": [[left, top, width, height]],
 }, 
 {
        "image_id": 100000,
        "score": 0.5012,
        "category_id": 6,
        "bbox": [
            60.99991,
            14.000025,
            39.00004,
            52.999981999999996
        ]
 }
]
```

### Reference
- [h5py - Quick Start Guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv5 實現目標檢測](https://tw511.com/a/01/29504.html)
- [2020_HW2](https://github.com/chia56028/Street-View-House-Numbers-Detection/blob/main/README.md)
