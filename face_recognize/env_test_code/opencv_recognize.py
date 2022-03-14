'''
Description: 使用opencv来进行人脸识别
Author: jiayuchen
Date: 2022-03-04 14:45:14
LastEditTime: 2022-03-14 14:58:18
IDE: VS code
'''

import cv2

class opencv_rec():
    def __init__(self): 
        CascadeClassifier_path = "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(
            CascadeClassifier_path
        )
        self.color = (0, 0, 255)
    
    def face_rec(self, img):
        color = self.color
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #调用人脸识别
        faceRects = self.classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(32, 32))
        print(f'Found {len(faceRects)} faces')
