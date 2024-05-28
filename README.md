# 基于Mini_Xception的实时人脸表情识别

论文[《Real-time Convolutional Neural Networks for Emotion and Gender Classification》](https://arxiv.org/pdf/1710.07557v1.pdf)

## 环境

- python 3.11
- opencv-python 4.9.0.80
- numpy 1.26.3
- pandas 2.2.2
- matplotlib 3.8.4
- pytorch 2.2.2

## 数据集

下载fer2013.csv并放至data/fer2013文件夹下，运行dataset_prepare.py

## 训练

运行train.py

## 图片测试

运行frame.py

## 摄像头测试

运行video.py

## 混淆矩阵

运行eval.py