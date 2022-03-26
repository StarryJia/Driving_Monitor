# -*- encoding: utf-8 -*-

'''
@File    :   Monitor.py
@Time    :   2022/03/26 10:48:12
@Author  :   StarryJia 
@Version :   1.0
@IDE     :   VS Code
'''
import sys

import cv2

sys.path.append('.\Monitor_warning')
# import yawn_and_wink.yawn_wink_dozeoff_no_graphical_interface as yw
import Monitor_warning.pose_estimation as pe

def main():
    print('[INFO] Monitor is activated.')
    cap = cv2.VideoCapture(0)
    pe.main(cap)

if __name__ == '__main__':
    main()