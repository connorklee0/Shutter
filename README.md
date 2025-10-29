# Directing Shutter with Subject Gestures

Justin Chang, Connor Lee, Anna Zhang

##### Table of Contents  
* [Overview](#overview)  
* [Installation](#installation)  
* [Usage](#usage)   
* [Demo(s)](#demos)
* [File Breakdown](#file-breakdown) 
* [Acknowledgements](#acknowledgements)

<a name="overview"/>

# Overview

The long tradition of self-portraiture. The death of realism in painting. The birth of the selfie. Taking a picture of yourself is something unique and special: the moment the subject and the photographer are one and the same.
 
Robot photographers hold tremendous promise in enhancing the quality, precision, and artistic appeal of images (O’Leary, 2019), but many current approaches take minimal user input. Existing models either choose from a list of fixed poses (Zhang, 2019), maximize an aesthetic metric learned from a limited dataset (AlZayer, 2021), or balance facial orientation with perceived emotions (Newbury, 2020). Therefore, the applicability of robot photographers to interactive subject-directed photography has yet to be extensively explored.
 
Professional photographers frequently use gestures in directing subjects, from encouraging rotations ("tilt your head to the left!" accompanied by rotating their hand) to subtle translations ("move your hand a little down!" accompanied by a 'patting' motion). Inspired by this approach, we aim to extend existing work with the robot photographer Shutter and design a programmatic pipeline for directing Shutter with subject gestures (Adamson, 2020).

<a name="installation"/>

# Installation

The following packages are required:

* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later

You can install these packages through navigating to the project directory and running

```bash
pip install -r requirements.txt
```

<a name="usage"/>

# Usage

After cloning the repository, installing dependencies, and navigating to the repository location, please follow the steps below. Note that these steps recapitulate aliases.sh:
```bash
# begin by bringing up Shutter
roslaunch shutter_bringup shutter_with_face.launch

# then turn on the controllers and run hand_tracking.py (need another terminal)
cd hand-gesture-recognition-mediapipe-main
rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_controller']
stop_controllers: ['']
strictness: 0 
start_asap: false
timeout: 0.0"
python3 hand_tracking.py

# launch the kinect
roslaunch azure_kinect_ros_driver driver.launch depth_mode:=WFOV_2X2BINNED body_tracking_enabled:=true color_enabled:=true 
```

<a name="demos"/>

# Demo(s)

These video(s) may take a few seconds to load!

![](demo1.gif)
<br><br>

<a name="file-breakdown"/>

# File Breakdown

Our contributions center on the following files:

1. <b>hand-gesture-recognition-mediapipe-main/hand_tracking.py:</b> Initializes a ROS node that takes in input from Shutter, the Azure Kinect, and the D435i RealSense camera. The input is used to recognize the subject's gesture and correspondingly direct Shutter to move. Our approach yields stable frame-limited performance by integratively interpreting the latest data from each input, using MediaPipe machine learning to classify the inputs, and directing Shutter to move through joint control. Currently, an open hand will cause Shutter to stand up, a thumbs up will cause Shutter to crouch and look upwards, and any other gesture (including the absence of a gesture) will place Shutter into a passive state.
2. <b>aliases.sh:</b> Useful bash commands for running our project alongside Shutter, the Azure Kinect, and the D435i RealSense camera.
3. <b>hand-gesture-recognition-mediapipe-main/model/keypoint_classifier/keypoint_classifier.hdf5</b>: The trained model for recognizing gestures in keypoints (single images).
4. <b>hand-gesture-recognition-mediapipe-main/model/point_history_classifier/point_history_classifier.hdf5</b>: The trained model for recognizing gestures from point histories (collections of images from past keypoint classifications).
5. <b>shutter_tester.py:</b> Code used to experiment with Shutter-ROS, but not part of the gesture recognition pipeline.
6. <b>shutter_hand_classification/src/gesture_recognition.py:</b> Code used to experiment with running MediaPipe, but not part of the gesture recognition pipeline.
7. <b>shutter_hand_classification/images/*:</b> Images generated from running the pipeline. Used to benchmark the efficiency of hand_tracking.py by evaluating how quickly saved images reflect real-world actions.

<a name="acknowledgements"/>

# Acknowledgements

We thank Dr. Marynel Vasquez for providing the course "Building Interactive Machines", Shutter robots, and mentorship. We thank Sasha Lew for his expert assistance and suggestions. We thank Google for developing [MediaPipe](https://github.com/google/mediapipe), Kazuhito Takahashi for his work in [applying MediaPipe to gesture recognition](https://github.com/kinivi/hand-gesture-recognition-mediapipe), and Nikita Kiselov for his work in translation.
