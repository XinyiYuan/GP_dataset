# GP_dataset

用于毕业设计模型训练的数据集

## 数据来源

Faceforensics

包含：

三百多个专人演员拍摄的影片（真实）

从YouTube上下载的约一千个影片（真实）

通过各种伪造技术生成的约三千个视频（伪造）

## 提取landmarks

使用mediapipe方法提取人脸三维特征（468维）

## 文件夹格式

-fake_videos 存储约三千个伪造视频

-fake_videos_txt 从fake_videos中提取出的landmarks

-real_videos_actor

-real_videos_actor_txt

-real_videos_youtube

-real_videos_youtube_txt

-calib_utils.py

-extract_landmarks_for_dataset.py

-landmark_utils.py

## 运行方式

将需要提取landmark的视频存放在一个文件夹中，并在extract_landmarks_for_dataset.py中更改文件夹路径。

'''
python extract_landmarks_for_dataset.py
'''
