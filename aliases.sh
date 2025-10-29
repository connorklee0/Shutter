# Assumptions:
# - you are in the repo folder
#       e.g. 'cd /home/jjc257/catkin_ws/src/jjc257-cpsc459-g2'
# - you have sourced the setup script of a catkin workspace
#       e.g. 'source ~/catkin_ws/devel/setup.bash'
# - you have installed all requirements
#       e.g. 'pip install -r requirements.txt'

# ----------------------------
# ONE COMMAND TO RULE THEM ALL
# ----------------------------

tmux \
  new-session "sleep 20; roslaunch hand-gesture-recognition-mediapipe-main shutter_gesture_recognition.launch" \; \
  split-window "roslaunch realsense2_camera rs_camera.launch initial_reset:=true"

# -----------------------------
# LAUNCH SHUTTER WITH LESS CMDS
# -----------------------------

roslaunch realsense2_camera rs_camera.launch initial_reset:=true
roslaunch hand-gesture-recognition-mediapipe-main shutter_gesture_recognition.launch

# then turn on the controllers and run hand_tracking.py (need another terminal)
cd hand-gesture-recognition-mediapipe-main
rosservice call /controller_manager/switch_controller "start_controllers: ['joint_group_controller'] stop_controllers: [''] strictness: 0  start_asap: false timeout: 0.0"
python3 hand_tracking.py

# -----------------------
# LAUNCH SHUTTER GESTURES
# -----------------------

# (optional) launch the realsense camera
# note: must be done FIRST!
roslaunch realsense2_camera rs_camera.launch initial_reset:=true

# launch the kinect
roslaunch azure_kinect_ros_driver driver.launch depth_mode:=WFOV_2X2BINNED body_tracking_enabled:=true color_enabled:=true 

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

# ---------------------
# OTHER USEFUL COMMANDS
# ---------------------

# launch rviz
rosrun rviz rviz

# show the rqt graph
rosrun rqt_graph rqt_graph

# publish to a joint
rostopic pub -1 /joint_group_controller/command std_msgs/Float64MultiArray "data: [-0.1, -1, -1.0, 0.05]"

# make shutter move
roslaunch shutter_moveit_config demo.launch moveit_controller_manager:=ros_control

# echo the current joint states of shutter
rostopic echo /joint_states

# TBD: one launch file (ask about tmux on the slack)
# rospy proxy service call ????
# train from kinect? train more data, from a more diverse grp of people
# joint safety, more collision?
# code cleanup, polish the rviz

# 800 x 480
# rqt image view
# move the window off once youre done viewing