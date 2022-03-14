'''
Description: 视频识别人脸环境测试
Author: jiayuchen
Date: 2022-03-04 14:30:21
LastEditTime: 2022-03-14 18:57:10
IDE: VS code
'''
import cv2

from opencv_recognize import opencv_rec

cap = cv2.VideoCapture(0)
while (1): 
    ret, img = cap.read()
    opencv_rec().face_rec(img)
cap.release()  # 释放摄像头