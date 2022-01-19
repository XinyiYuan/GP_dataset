# GP_dataset

用于毕业设计模型训练的数据集

## 数据来源

Faceforensics

包含：

从YouTube上下载的约一千个影片（真实）

通过DF伪造技术生成的约一千个视频（伪造）

## 提取landmarks

使用mediapipe方法提取人脸三维特征（468维）


## 运行方式

将需要提取landmark的视频存放在一个文件夹中，并在extract_landmarks_for_dataset.py中更改文件夹路径。

'''
python extract_landmarks_for_dataset.py
'''
