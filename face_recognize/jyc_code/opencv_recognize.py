'''
Description: 使用opencv来进行人脸识别
Author: jiayuchen
Date: 2022-03-04 14:45:14
LastEditTime: 2022-03-04 15:51:35
IDE: VS code
'''

import cv2

class opencv_rec():
    def __init__(self): 
        CascadeClassifier_path = "../../../opencv/data/haarcascades/\
haarcascade_frontalface_default.xml"
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
        if len(faceRects): 
            for face in faceRects:
                x, y, w, h = face
                cv2.rectangle(img, (x, y), (x + h,y + w), color, 2)
                # 框出人脸
                cv2.circle(img, (x + w // 4, y + h // 4 + 30),
                min(w // 8, h // 8), color)
                cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), 
                min(w // 8, h // 8), color)
                # 框出双眼
                cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                      (x + 5 * w // 8, y + 7 * h // 8), color)
                # 框出嘴巴
        cv2.imshow("Image", img)
