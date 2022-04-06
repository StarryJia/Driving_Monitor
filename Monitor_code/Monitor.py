# -*- encoding: utf-8 -*-

'''
@File    :   Monitor.py
@Time    :   2022/03/26 10:48:12
@Author  :   StarryJia 
@Version :   1.0
@IDE     :   VS Code
'''
from concurrent.futures import thread
import sys
import cv2
import threading
import os
import time
import shutil
import global_variable

# sys.path.append(r"../")
from yolov5.detect import parse_opt as yolo_parse_opt
from yolov5.detect import main as yolo_main



class yolo_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        while True:
            if not os.listdir(r"./yolov5/data/images"):
                print("空文件夹!")
                time.sleep(1)
            else: 
                time.sleep(1)
                yolo_main(yolo_parse_opt())

                # 读取txt: 类别, 置信度
                if os.listdir(r"./yolov5/runs/detect/exp/labels"): #文件夹不为空, 检测出手机或者吸烟
                    with open(r"./yolov5/runs/detect/exp/labels/please_test.txt") as fp:
                        for line in fp.readlines():
                            line = line.strip('\n')  #去掉列表中每一个元素的换行符
                            line = line.split(" ")
                            print(line)
                            category = "cigarette" if line[0] == '1' else "cellphone" #类别  
                            print((category + "    ") * 3)
                            global_variable.set_dic_value(category, float(line[5]))#设置置信度
                            print("line[5]         " + line[5])


                # 删除文件夹D:\\workspace\\001Safe-Driving-Monitoring-System\jiehe\\yolov5\\runs\\detect\\exp, 以及之下的所有文件和文件夹
                shutil.rmtree(path=r"./yolov5/runs/detect/exp")

                # 删除图片
                os.remove(r"./yolov5/data/images/please_test.jpg")

                # 设置has_sent=0
                global_variable.set_sent_value(0)


        



sys.path.append(r'./Monitor_warning')
# import yawn_and_wink.yawn_wink_dozeoff_no_graphical_interface as yw
import Monitor_warning.pose_estimation as pe

def main():

    global_variable._init()

    print('[INFO] Monitor is activated.')
    yolo = yolo_thread()
    yolo.start()

    cap = cv2.VideoCapture(0)
    pe.main(cap)

if __name__ == '__main__':
    main()