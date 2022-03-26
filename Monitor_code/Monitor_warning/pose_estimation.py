# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run pose classification and pose estimation."""
import argparse
import logging
import sys
import time
import math

from scipy.spatial import distance as dist
#from imutils.video import FileVideoStream
#from imutils.video import VideoStream
from imutils import face_utils
import numpy as np # 数据处理的库 numpy
#import argparse
import imutils
import time
import dlib
import cv2
from cv2 import sqrt
from ml import Classifier
from ml import Movenet
from ml import MoveNetMultiPose
from ml import Posenet
import utils

def dis(Acoord,Bcoord):
  x = Acoord.x - Bcoord.x
  y = Acoord.y - Bcoord.y
  return math.sqrt((x**2) + (y**2))
  
def run(estimation_model: str, tracker_type: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int, cap) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    estimation_model: Name of the TFLite pose estimation model.
    tracker_type: Type of Tracker('keypoint' or 'bounding_box').
    classification_model: Name of the TFLite pose classification model.
      (Optional)
    label_file: Path to the label file for the pose classification model. Class
      names are listed one name per line, in the same order as in the
      classification model output. See an example in the yoga_labels.txt file.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Notify users that tracker is only enabled for MoveNet MultiPose model.
  if tracker_type and (estimation_model != 'movenet_multipose'):
    logging.warning(
        'No tracker will be used as tracker can only be enabled for '
        'MoveNet MultiPose model.')

  # Initialize the pose estimator selected.
  if estimation_model in ['movenet_lightning', 'movenet_thunder']:
    pose_detector = Movenet(estimation_model)
  elif estimation_model == 'posenet':
    pose_detector = Posenet(estimation_model)
  elif estimation_model == 'movenet_multipose':
    pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
  else:
    sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  classification_results_to_show = 3
  fps_avg_frame_count = 10
  keypoint_detection_threshold_for_classifier = 0.1
  classifier = None
  status = {
  'legal': 0,
  'warning': 1,
  'danger': 2,
}
  flag = 'legal'
  warning_num = 0
  legal_num = 0
  continuous_nums = 100
  legal_continuous_nums = 20
  legal_start_num = 0

  # Initialize the classification model
  if classification_model:
    classifier = Classifier(classification_model, label_file)
    classification_results_to_show = min(classification_results_to_show,
                                         len(classifier.pose_class_names))

  # Continuously capture images from the camera and run inference
  while True:
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    yawn_wink_dozeoff(image)
    counter += 1
    image = cv2.flip(image, 1)

    if estimation_model == 'movenet_multipose':
      # Run pose estimation using a MultiPose model.
      list_persons = pose_detector.detect(image)
    else:
      # Run pose estimation using a SinglePose model, and wrap the result in an
      # array.
      list_persons = [pose_detector.detect(image)]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, list_persons)
    '''
    L_EAR is right ear in the image
    R_EAR is left ear in the image
    L_WRIST is right wrist in the image
    R_WRIST is left wrist in the image
    '''
    L_WRIST = list_persons[0].keypoints[9].coordinate
    R_WRIST = list_persons[0].keypoints[10].coordinate
    L_EAR = list_persons[0].keypoints[3].coordinate
    R_EAR = list_persons[0].keypoints[4].coordinate
    left = dis(L_WRIST,L_EAR)
    right = dis(R_WRIST,R_EAR)
    left_to_right = dis(L_WRIST,R_EAR)
    right_to_left = dis(R_WRIST,L_EAR)

    if (left < 140) or (right < 140) or (left_to_right < 180) or (right_to_left < 180):
      # print('not ok')
      if status[flag] == 0:
        # 如果第一次检测到手的位置异常，进入警戒模式，并记录当前是第几帧
        flag = 'warning'
        start_num = counter
        warning_num += 1
      if status[flag] == 1:
        # 警戒模式下每检测到一次手的位置异常则增加位置异常数
        warning_num += 1
    elif status[flag] != 0:
      # 如果状态不合法但是手部状态正常
      if (counter - legal_start_num) > (legal_continuous_nums + 50):
        legal_start_num = counter
        legal_num = 0
      if (counter - legal_start_num) > legal_continuous_nums and legal_num >= 0.8 * (counter - legal_start_num):
        # 如果正常状态超过至少连续legal_continuous_nums帧数的一定比例
        flag = 'legal'
        print(f'[MODE CHANGE] {flag}')
        warning_num = 0
        legal_num = 0
        start_num = 0
      else:
        legal_num += 1
        # print(legal_num)
    # else:
    #   print('ok')
        
    
    if status[flag] == 1:
      #如果状态是警戒模式
      if (counter - start_num) >= continuous_nums:
          # 连续检查continuous_nums帧
          if warning_num > 0.9 * continuous_nums:
            # 如果90%的情况都检测到手的位置异常，则进入危险驾驶模式
            flag = 'danger'
            print(f'[MODE CHANGE] {flag}')
    if status[flag] == 2:
      # 如果进入了危险模式，持续提醒
      print('[WARNING] detected using moble phone!!!!!!')

    if classifier:
      # Check if all keypoints are detected before running the classifier.
      # If there's a keypoint below the threshold, show an error.
      person = list_persons[0]
      min_score = min([keypoint.score for keypoint in person.keypoints])
      if min_score < keypoint_detection_threshold_for_classifier:
        error_text = 'Some keypoints are not detected.'
        print(error_text)
        # text_location = (left_margin, 2 * row_size)
        # cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
        #             font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible in the camera.'
        print(error_text)
        # text_location = (left_margin, 3 * row_size)
        # cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
        #             font_size, text_color, font_thickness)
      else:
        # Run pose classification
        prob_list = classifier.classify_pose(person)

        # Show classification results on the image
        for i in range(classification_results_to_show):
          class_name = prob_list[i].label
          probability = round(prob_list[i].score, 2)
          result_text = class_name + ' (' + str(probability) + ')'
          print(result_text)
          # text_location = (left_margin, (i + 2) * row_size)
          # cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
          #             font_size, text_color, font_thickness)
    # Calculate the FPS
    # if counter % fps_avg_frame_count == 0:
    #   end_time = time.time()
    #   fps = fps_avg_frame_count / (end_time - start_time)
    #   start_time = time.time()

    # # Show the FPS
    # fps_text = 'FPS = ' + str(int(fps))
    # print(fps_text)
    # text_location = (left_margin, row_size)
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    # if cv2.waitKey(1) == 27:
    #   break
    # cv2.imshow(estimation_model, image)

  cap.release()
  # cv2.destroyAllWindows()


def main(cap):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of estimation model.',
      required=False,
      default='movenet_lightning')
  parser.add_argument(
      '--tracker',
      help='Type of tracker to track poses across frames.',
      required=False,
      default='bounding_box')
  parser.add_argument(
      '--classifier', help='Name of classification model.', required=False)
  parser.add_argument(
      '--label_file',
      help='Label file for classification.',
      required=False,
      default='labels.txt')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=720)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, args.tracker, args.classifier, args.label_file,
      int(args.cameraId), args.frameWidth, args.frameHeight, cap)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                         [1.330353, 7.122144, 6.903745],  #29左眉右角
                         [-1.330353, 7.122144, 6.903745], #34右眉左角
                         [-6.825897, 6.760612, 4.402142], #38右眉右上角
                         [5.311432, 5.485328, 3.987654],  #13左眼左上角
                         [1.789930, 5.393625, 4.413414],  #17左眼右上角
                         [-1.789930, 5.393625, 4.413414], #25右眼左上角
                         [-5.311432, 5.485328, 3.987654], #21右眼右上角
                         [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                         [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                         [2.774015, -2.080775, 5.048531], #43嘴左上角
                         [-2.774015, -2.080775, 5.048531],#39嘴右上角
                         [0.000000, -3.116408, 6.097667], #45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])#6下巴角

# 相机坐标系(XYZ): 添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv): 相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy): 填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)



# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):# 头部姿态估计
    # （像素坐标集合）填写2D参考点, 注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵: 
    # rotation_vec表示旋转矩阵, translation_vec表示平移矩阵, cam_matrix与K矩阵对应, dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差: 原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t, 输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,dist_coeffs)
    #reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示
    reprojectdst = reprojectdst.reshape(8, 2).astype(int)

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)#罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))# 水平拼接, vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
 
 
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle# 投影误差, 欧拉角

def eye_aspect_ratio(eye):
    # 垂直眼标志（X, Y）坐标
    A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X, Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear
 
def mouth_aspect_ratio(mouth):# 嘴部
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# 定义常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
# 瞌睡点头
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0

# 初始化DLIB的人脸检测器（HOG）, 然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步: 使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步: 使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
# 第三步: 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def yawn_wink_dozeoff(frame):
  global EYE_AR_THRESH
  global EYE_AR_CONSEC_FRAMES
  global MAR_THRESH
  global MOUTH_AR_CONSEC_FRAMES
  global HAR_THRESH
  global NOD_AR_CONSEC_FRAMES
  global COUNTER
  global TOTAL
  global mCOUNTER
  global mTOTAL
  global hCOUNTER
  global hTOTAL
  '''
  Begin monitoring yawn, wink and dozeoff
  '''
  # 第五步: 进行循环, 读取图片, 并对图片做维度扩大, 并进灰度化
  frame = imutils.resize(frame, width=720)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # 第六步: 使用detector(gray, 0) 进行脸部位置检测
  rects = detector(gray, 0)
  
  # 第七步: 循环脸部位置信息, 使用predictor(gray, rect)获得脸部特征位置的信息
  for rect in rects:
      shape = predictor(gray, rect)
      
      # 第八步: 将脸部特征信息转换为数组array的格式
      shape = face_utils.shape_to_np(shape)
      
      # 第九步: 提取左眼和右眼坐标
      leftEye = shape[lStart:lEnd]
      rightEye = shape[rStart:rEnd]
      # 嘴巴坐标
      mouth = shape[mStart:mEnd]        
      
      # 第十步: 构造函数计算左右眼的EAR值, 使用平均值作为最终的EAR
      leftEAR = eye_aspect_ratio(leftEye)
      rightEAR = eye_aspect_ratio(rightEye)
      ear = (leftEAR + rightEAR) / 2.0
      # 打哈欠
      mar = mouth_aspect_ratio(mouth)

      # 第十一步: 使用cv2.convexHull获得凸包位置, 使用drawContours画出轮廓位置进行画图操作
      #leftEyeHull = cv2.convexHull(leftEye)
      #rightEyeHull = cv2.convexHull(rightEye)
      #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
      #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
      #mouthHull = cv2.convexHull(mouth)
      #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

      # 第十二步: 进行画图操作, 用矩形框标注人脸
      #left = rect.left()
      #top = rect.top()
      #right = rect.right()
      #bottom = rect.bottom()
      #cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)    

      '''
          分别计算左眼和右眼的评分求平均作为最终的评分, 如果小于阈值, 则加1, 如果连续3次都小于阈值, 则表示进行了一次眨眼活动
      '''
      # 第十三步: 循环, 满足条件的, 眨眼次数+1
      if ear < EYE_AR_THRESH:# 眼睛长宽比: 0.2
          COUNTER += 1
      
      else:
          # 如果连续3次都小于阈值, 则表示进行了一次眨眼活动
          if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值: 3
              TOTAL += 1
              print("Blinks: {}".format(TOTAL))       #眨眼次数
          # 重置眼帧计数器
          COUNTER = 0
          
      # 第十四步: 进行画图操作, 同时使用cv2.putText将眨眼次数进行显示
      #cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  #脸的个数     
      #cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 眼帧计数器
      #cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)      #EAR值
      #cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)     #眨眼次数

      # print("Faces: {}".format(len(rects)))   #脸的个数
      # print("COUNTER: {}".format(COUNTER))    # 眼帧计数器
      # print("EAR: {:.2f}".format(ear))        #EAR值
      
      
      '''
          计算张嘴评分, 如果小于阈值, 则加1, 如果连续3次都小于阈值, 则表示打了一次哈欠, 同一次哈欠大约在3帧
      '''
      # 同理, 判断是否打哈欠    
      if mar > MAR_THRESH:# 张嘴阈值0.5
          mCOUNTER += 1
          #cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
          print("Yawning!")
      else:
          # 如果连续3次都小于阈值, 则表示打了一次哈欠
          if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值: 3
              mTOTAL += 1
              print("Yawning: {}".format(mTOTAL))     # 打哈欠次数
          # 重置嘴帧计数器
          mCOUNTER = 0
      #cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     # 嘴帧计数器
      #cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      #cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)       #打哈欠次数
      # print("COUNTER: {}".format(mCOUNTER))   # 嘴帧计数器
      # print("MAR: {:.2f}".format(mar))        # MAR
      
      
      """
      瞌睡点头
      """
      # 第十五步: 获取头部姿态
      reprojectdst, euler_angle = get_head_pose(shape)
      
      har = euler_angle[0, 0]# 取pitch旋转角度
      if har > HAR_THRESH:# 点头阈值0.3
          hCOUNTER += 1
      else:
          # 如果连续3次都小于阈值, 则表示瞌睡点头一次
          if hCOUNTER >= NOD_AR_CONSEC_FRAMES:# 阈值: 3
              hTOTAL += 1
              print("Nod: {}".format(hTOTAL))
          # 重置点头帧计数器
          hCOUNTER = 0
      
      # 绘制正方体12轴
      #for start, end in line_pairs:
      #    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
      # 显示角度结果
      #cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), thickness=2)# GREEN
      #cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), thickness=2)# BLUE
      #cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)# RED    
      #cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
      # print("X: " + "{:7.2f}".format(euler_angle[0, 0]))
      # print("Y: " + "{:7.2f}".format(euler_angle[1, 0]))
      # print("Z: " + "{:7.2f}".format(euler_angle[2, 0]))
      

      # 第十六步: 进行画图操作, 68个特征点标识
      #for (x, y) in shape:
      #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

      #print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴: "+str([False,True][mar > MAR_THRESH]))
      #print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼: "+str([False,True][COUNTER>=1]))
  
  # 确定疲劳提示:眨眼50次, 打哈欠15次, 瞌睡点头15次
  if TOTAL >= 50 or mTOTAL>=15 or hTOTAL>=15:
      #cv2.putText(frame, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
      print("SLEEP!!!")
      
  # 按q退出
  #cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
  # 窗口显示 show with opencv
  #cv2.imshow("Frame", frame)

  # if the `q` key was pressed, break from the loop
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #    break
  '''
  退出循环这里可以从外界传过来一个消息然后退出循环
  '''

if __name__ == '__main__':
  main()
