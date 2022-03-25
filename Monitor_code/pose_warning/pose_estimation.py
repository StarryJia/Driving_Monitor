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
        label_file: str, camera_id: int, width: int, height: int) -> None:
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
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

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
        print('in legal')
        flag = 'legal'
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
            print('in danger!!!!')
    if status[flag] == 2:
      # 如果进入了危险模式，持续提醒
      print('dangerous!!!!')

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


def main():
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
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()
