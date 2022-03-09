'''
Description: 视频识别人脸
Author: jiayuchen
Date: 2022-03-04 14:30:21
LastEditTime: 2022-03-04 15:35:44
IDE: VS code
'''
import cv2

from opencv_recognize import opencv_rec

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while (1): 
    ret, img = cap.read()
    opencv_rec().face_rec(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源