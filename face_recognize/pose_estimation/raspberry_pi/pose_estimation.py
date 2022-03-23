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
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
    # print(list_persons)
    """
    [Person(keypoints=[
    KeyPoint(body_part=<BodyPart.NOSE: 0>, coordinate=Point(x=336, y=272), score=0.5676351), 
    KeyPoint(body_part=<BodyPart.LEFT_EYE: 1>, coordinate=Point(x=379, y=230), score=0.6834916), 
    KeyPoint(body_part=<BodyPart.RIGHT_EYE: 2>, coordinate=Point(x=295, y=224), score=0.61684793), 
    KeyPoint(body_part=<BodyPart.LEFT_EAR: 3>, coordinate=Point(x=427, y=257), score=0.438448), 
    KeyPoint(body_part=<BodyPart.RIGHT_EAR: 4>, coordinate=Point(x=242, y=246), score=0.8014671), 
    KeyPoint(body_part=<BodyPart.LEFT_SHOULDER: 5>, coordinate=Point(x=495, y=420), score=0.6528228), 
    KeyPoint(body_part=<BodyPart.RIGHT_SHOULDER: 6>, coordinate=Point(x=140, y=428), score=0.6682893), 
    KeyPoint(body_part=<BodyPart.LEFT_ELBOW: 7>, coordinate=Point(x=569, y=475), score=0.3175025), 
    KeyPoint(body_part=<BodyPart.RIGHT_ELBOW: 8>, coordinate=Point(x=89, y=487), score=0.07942194), 
    KeyPoint(body_part=<BodyPart.LEFT_WRIST: 9>, coordinate=Point(x=576, y=464), score=0.2471753), 
    KeyPoint(body_part=<BodyPart.RIGHT_WRIST: 10>, coordinate=Point(x=317, y=460), score=0.008905202), 
    KeyPoint(body_part=<BodyPart.LEFT_HIP: 11>, coordinate=Point(x=487, y=505), score=0.10334611), 
    KeyPoint(body_part=<BodyPart.RIGHT_HIP: 12>, coordinate=Point(x=156, y=507), score=0.047061205), 
    KeyPoint(body_part=<BodyPart.LEFT_KNEE: 13>, coordinate=Point(x=535, y=469), score=0.13279632), 
    KeyPoint(body_part=<BodyPart.RIGHT_KNEE: 14>, coordinate=Point(x=113, y=460), score=0.2163659), 
    KeyPoint(body_part=<BodyPart.LEFT_ANKLE: 15>, coordinate=Point(x=534, y=189), score=0.12815252), 
    KeyPoint(body_part=<BodyPart.RIGHT_ANKLE: 16>, coordinate=Point(x=181, y=453), score=0.010690182)], 
    bounding_box=Rectangle(start_point=Point(x=89, y=189), end_point=Point(x=576, y=507)), score=0.4287954, id=None)]
    """
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
    # print('L_W = ',L_WRIST)
    # print('L_E = ',L_EAR)
    # print('left = ',dis(L_WRIST,L_EAR))
    # print('L_W = ',R_WRIST)
    # print('L_E = ',R_EAR)
    # print('right = ',dis(R_WRIST,R_EAR))
    left = dis(L_WRIST,L_EAR)
    right = dis(R_WRIST,R_EAR)
    left_to_right = dis(L_WRIST,R_EAR)
    right_to_left = dis(R_WRIST,L_EAR)
    if (left < 140) or (right < 140) or (left_to_right < 180) or (right_to_left < 180):
      print('warning')
    else:
      print('ok')

    if classifier:
      # Check if all keypoints are detected before running the classifier.
      # If there's a keypoint below the threshold, show an error.
      person = list_persons[0]
      min_score = min([keypoint.score for keypoint in person.keypoints])
      if min_score < keypoint_detection_threshold_for_classifier:
        error_text = 'Some keypoints are not detected.'
        text_location = (left_margin, 2 * row_size)
        cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible in the camera.'
        text_location = (left_margin, 3 * row_size)
        cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
      else:
        # Run pose classification
        prob_list = classifier.classify_pose(person)

        # Show classification results on the image
        for i in range(classification_results_to_show):
          class_name = prob_list[i].label
          probability = round(prob_list[i].score, 2)
          result_text = class_name + ' (' + str(probability) + ')'
          text_location = (left_margin, (i + 2) * row_size)
          cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)
    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow(estimation_model, image)

  cap.release()
  cv2.destroyAllWindows()


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
