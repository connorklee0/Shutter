#!/usr/bin/env python3   
import sys
import os
import rospy
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray
from urdf_parser_py.urdf import URDF
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy.optimize import least_squares
import tf2_geometry_msgs
import tf.transformations as tft

# local imports
import app
from utils.movement_helpers import transform_msg_to_T, \
    make_joint_rotation, target_in_camera_frame, warn_msg, \
        filter_markers_by_type, clamp_joint_angle, K4ABT_JOINT_HEAD

# constants 
UPDATE_FPS = 240 # number of times to update per second
UPDATE_DUR = 1.0 / UPDATE_FPS # duration between updates in seconds
REPORT_PER = 1200 # report info every X amount of updates
GESTURE_RATE = 20 # print gesture every X amount of attempts
CONTROLLERS_ON = "rosservice call /controller_manager/switch_controller \"start_controllers: ['joint_group_controller']\nstop_controllers: ['']\nstrictness: 0\nstart_asap: false\ntimeout: 0.0\""

class HandTrackingNode():
    def __init__(self):
        rospy.init_node('hand_tracking', anonymous=True)
        rospy.wait_for_service('/controller_manager/switch_controller')
        project_path = rospy.get_param("~project_path", default=os.getcwd())
        os.chdir(project_path)
        os.system(CONTROLLERS_ON)
        rospy.on_shutdown(self.cleanup)

        # used to make shutter look at you
        self.base_link = "base_link"
        self.biceps_link = "biceps_link"
        self.camera_link = "camera_color_optical_frame"

        self.robot = URDF.from_parameter_server() # URDF model of the robot
        self.moving_joints = self.populate_moving_joints() # gets specs of all moving joints
        self.ideal_joints = None # ideal position of the robot
        self.rs_image = None # realsense image
        self.ka_image = None # kinect image
        self.ka_track = None # kinect body tracking
        self.joint_states = None # current joint states
        self.last_gesture = -1 # last detected gesture
        self.last_markers = None # last detected markers
        self.update_count = 0 # number of updates
        self.last_update_time = rospy.Time.now() # current time
        self.bridge = CvBridge() # convert to cv2 images

        # subscribe to realsense camera image
        rospy.Subscriber("/camera/color/image_raw", Image, self.rs_image_callback, queue_size=5)
        # subscribe to kinect camera
        rospy.Subscriber("/rgb/image_raw", Image, self.ka_image_callback, queue_size=5)
        # subscribe to kinect tracking
        rospy.Subscriber("/body_tracking_data", MarkerArray, self.ka_track_callback, queue_size=5)
        # subscribe to joint states
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback, queue_size=5)

        # create joint publisher
        self.joint_pub = rospy.Publisher("/joint_group_controller/command", Float64MultiArray, queue_size=5)
        # create debug image publisher
        self.debug_image_pub = rospy.Publisher('/hand_tracking/debug_image', Image, queue_size=5)
        # create shutter image publisher
        self.photo_image_pub = rospy.Publisher('/hand_tracking/photo_image', Image, queue_size=5)
        # create filtered markerarray of joints
        self.filtered_ma_pub = rospy.Publisher('/hand_tracking/filtered_ma', MarkerArray, queue_size=5)

        # create transform buffer, listener, broadcaster (likely not needed)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        # try to prevent race conditions
        self.updating_gesture = False
        self.updating = False

        # store the current gesture and how many times you've been on that gesture
        self.gesture_id = -1
        self.gesture_counter = 0

        # initializes all necessary values for media pipe classification
        self.initialize_media_pipe()

        # keep the node running indefinitely
        rospy.spin()

    def initialize_media_pipe(self):
        """
        Helper function to initialize all of the necessary values to set up the media pipe library and gesture classifcation
        """
        # ---------------------------------
        # INITIALIZE MEDIAPIPE FOR GESTURES
        # ---------------------------------
        # note: code was modified from app.py main() function
        use_static_image_mode = True
        min_detection_confidence = 0.7
        min_tracking_confidence = 0.5

        self.use_brect = True

        mp_hands = app.mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = app.KeyPointClassifier()
        self.point_history_classifier = app.PointHistoryClassifier()

        # get labels files
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = app.csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = app.csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]

        # create structures for recording performance
        self.cvFpsCalc = app.CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = app.deque(maxlen=self.history_length)
        self.finger_gesture_history = app.deque(maxlen=self.history_length)
        self.mode = 0

    def get_p_T1_T2(self, msg):
        """
        Helper function for compute_joints_position()
        :param msg: target message
        :return: target in baselink, transform from base_link to biceps, transform from biceps to camera
        """

        # the marker is initially in a depth_camera_link frame, so change that to camera_base
        try:
            tf_special = self.tf_buffer.lookup_transform(self.biceps_link,
                                                        self.base_link,  # source frame
                                                        msg.header.stamp,
                                                        # get the transform at the time the pose was generated
                                                        rospy.Duration(1.0)) 
        except:
            return None, None, None
        tf_special.header = msg.header
        tf_special.header.frame_id = "camera_base"
        tf_special.child_frame_id = "depth_camera_link"
        tf_special.transform.translation.x = 0.0
        tf_special.transform.translation.y = 0.0
        tf_special.transform.translation.z = 0.0017999999690800905
        tf_special.transform.rotation.x = 0.525482745498759
        tf_special.transform.rotation.y = -0.5254827454987588
        tf_special.transform.rotation.z = 0.473146789255815
        tf_special.transform.rotation.w = -0.4731467892558148
 
        pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, tf_special)

        p = [pose_transformed.pose.position.x - 0.1,
             pose_transformed.pose.position.y,
             pose_transformed.pose.position.z + 0.7] 

        # get transform from base link to camera link (base_link -> biceps_link and biceps_link -> camera_link)
        try:
            transform = self.tf_buffer.lookup_transform(self.biceps_link,
                                                        self.base_link,  # source frame
                                                        msg.header.stamp,
                                                        # get the transform at the time the pose was generated
                                                        rospy.Duration(1.0))  # wait for 1 second
            T1 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to compute new position for the robot because {}".format(e))
            T1 = None

        try:
            transform = self.tf_buffer.lookup_transform(self.camera_link,
                                                        self.biceps_link,  # source frame
                                                        msg.header.stamp,
                                                        # get the transform at the time the pose was generated
                                                        rospy.Duration(1.0))  # wait for 1 second
            T2 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(e)
            T2 = None

        return p, T1, T2

    def compute_joints_position(self, msg, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: new joint positions for joint1 and joint3; or None if something went wrong
        """
        p, T1, T2 = self.get_p_T1_T2(msg)
        if p is None or T1 is None or T2 is None:
            return None

        # compute the required motion for the robot using black-box optimization
        x0 = [-np.arctan2(p[1], p[0]), 0.0]
        res = least_squares(target_in_camera_frame, x0,
                            bounds=([-np.pi, -np.pi * 0.5], [np.pi, np.pi * 0.5]),
                            args=(p, self.robot.joints[1].axis, self.robot.joints[3].axis, T1, T2))
        # print("result: {}, cost: {}".format(res.x, res.cost))

        offset_1 = -res.x[0]
        offset_3 = -res.x[1]

        # note: we leave the safety clamping for later
        return joint1 + offset_1, joint3 + offset_3

    def populate_moving_joints(self):
        '''
        Creates a dictionary with the specifications of all moving joints.
        '''
        joints_dict = {}
        sr_joints = self.robot.joints
        for joint in sr_joints:
            if joint.type == "revolute":
                print("adding new mobile joint: {}".format(joint.name))
                # due to the kinect apparatus, we add additional restrictions
                if joint.name == 'joint_1':
                    joint.limit.lower = max(joint.limit.lower, -1.4)
                    joint.limit.upper = min(joint.limit.upper, 1.4)
                if joint.name == 'joint_2':
                    joint.limit.lower = max(joint.limit.lower, -1.5)
                    joint.limit.upper = min(joint.limit.upper, -1.2)
                if joint.name == 'joint_3':
                    joint.limit.upper = min(joint.limit.upper, 0.3)
                if joint.name == 'joint_4':
                    joint.limit.lower = max(joint.limit.lower, -0.25)
                joints_dict[joint.name] = joint
        return joints_dict

    def cv2_to_imgmsg(self, cv_image):
        '''
        Converts a cv2 image to an Image message
        :param cv_image: a cv2 object
        '''
        return self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

    def imgmsg_to_cv2(self, img_msg):
        '''
        Converts an Image message to a cv2 image
        :param img_msg: an Image message
        '''
        return self.bridge.imgmsg_to_cv2(img_msg)

    def rs_image_callback(self, rs_image):
        '''
        Assigns the newest rs_image and requests an update.
        :param rs_image: an Image msg
        '''
        if self.rs_image is None:
            print("Receiving rs_image data!")
        self.rs_image = rs_image
        self.fixed_rate_update()

    def rs_depth_callback(self, rs_depth):
        '''
        Assigns the newest rs_depth and requests an update. NOT USED
        :param rs_depth: an Image msg
        '''
        if self.rs_depth is None:
            print("Receiving rs_depth data!")
        self.rs_depth = self.imgmsg_to_cv2(rs_depth)
        self.fixed_rate_update()

    def ka_image_callback(self, ka_image):
        '''
        Assigns the newest ka_image and requests an update.
        :param ka_image: an Image msg
        '''
        if self.ka_image is None:
            print("Receiving ka_image data!")
        if self.updating_gesture:
            return
        self.ka_image = self.imgmsg_to_cv2(ka_image)
        self.gesture_update()

    def ka_depth_callback(self, ka_depth):
        '''
        Assigns the newest ka_depth and requests an update. NOT USED
        :param ka_depth: an Image msg
        '''
        if self.ka_depth is None:
            print("Receiving ka_depth data!")
        self.ka_depth = self.imgmsg_to_cv2(ka_depth)
        self.fixed_rate_update()

    def ka_track_callback(self, ka_track):
        '''
        Assigns the newest ka_track and requests an update.
        :param ka_track: a MarkerArray msg
        '''
        if self.ka_track is None:
            print("Receiving ka_track data!")
        self.ka_track = ka_track
        ka_filtered = app.copy.deepcopy(ka_track)
        ka_filtered.markers = filter_markers_by_type(ka_track.markers)
        self.filtered_ma_pub.publish(ka_filtered)
        self.fixed_rate_update()

    def joint_states_callback(self, joint_states):
        '''
        Assigns the newest joint_states and requests an update.
        :param joint_states: a JointState msg
        '''
        if self.joint_states is None:
            print("Receiving joint_states data!")
        self.joint_states = joint_states
        self.fixed_rate_update()

    def gesture_update(self):
        '''
        '''
        # note: code was modified from app.py main() function
        fps = self.cvFpsCalc.get()
        number, mode = -1, 0

        image = self.ka_image
        image = app.cv.flip(image, 1) # mirror the image
        debug_image = app.copy.deepcopy(image)

        image = app.cv.cvtColor(image, app.cv.COLOR_BGR2RGB) # convert to RGB

        image.flags.writeable = False # set image to read-only
        try:
            results = self.hands.process(image)
        except:
            self.updating = False
            return
        image.flags.writeable = True # restore ability to edit the image

        # if some kind of hand is identifiable ...
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # calculate bounding box
                brect = app.calc_bounding_rect(debug_image, hand_landmarks)
                # calculate landmarks
                landmark_list = app.calc_landmark_list(debug_image, hand_landmarks)

                # convert coordinates
                pre_processed_landmark_list = app.pre_process_landmark(
                    landmark_list
                )
                pre_processed_point_history_list = app.pre_process_point_history(
                    debug_image, 
                    self.point_history
                )
                # write to the dataset file
                app.logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # classify the current image's hand gesture
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                # note that the point gesture also has position data (where are you pointing to?)
                if hand_sign_id == 2: 
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                # classify the current image's finger gesture
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                # report the finger gesture as the mode of recent finger gestures
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = app.Counter(
                    self.finger_gesture_history).most_common()

                # draw on debug_image
                debug_image = app.draw_bounding_rect(self.use_brect, debug_image, brect)
                debug_image = app.draw_landmarks(debug_image, landmark_list)
                debug_image = app.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                self.last_gesture = hand_sign_id

        else:
            self.point_history.append([0, 0])
            self.last_gesture = -1

        debug_image = app.draw_point_history(debug_image, self.point_history)
        debug_image = app.draw_info(debug_image, fps, mode, number)
        self.debug_image_pub.publish(self.cv2_to_imgmsg(debug_image))

    def fixed_rate_update(self):
        '''
        Limits updates to a certain frequency per second.
        '''
        delta_time = rospy.Time.now() - self.last_update_time 
        if self.updating or delta_time.to_sec() < UPDATE_DUR:
            return
        self.update()
        self.update_count += 1 
        # account for the time it took to update
        self.last_update_time = rospy.Time.now()

    def save_cv_image(self, image_dir, title, cv_image):
        '''
        Saves a cv2 image to image_dir with the given title.
        :param image_dir: a valid directory
        :param title: a name for the picture (no extension)
        :param cv_image: the cv2 image
        '''
        addr = "{}/{}_update{}.jpeg".format(image_dir, title, self.update_count)
        cv2.imwrite(addr, cv_image)
        
    def update(self):
        '''
        Responds to gestures and prints reports at certain intervals.
        '''
        self.updating = True

        # reports / displays verbose info
        if self.update_count % REPORT_PER == 0:
            self.display_input_info()

        # useful for seeing if images are being updated over time
        # if self.update_count % 60 == 0 and self.update_count < 600:
        #     base_path = "/home/jjc257/catkin_ws/src/jjc257-cpsc459-g2/shutter_hand_classification/images"
        #     self.save_cv_image(base_path, "ka_image", self.ka_image)
        
        # start shutter in a neutral position
        if self.update_count < 60:  
            self.move_joints(0, -1.2, -1, 0.0, update_ideal=True)
        else:
            self.move_shutter_by_gesture(self.last_gesture)

        # demonstrates joint safety
        # if self.update_count % 240 == 0:
        #     self.move_joints(-5, None, -4, -4)
        # if self.update_count % 240 == 60:
        #     self.move_joints(5, None, -4, -4)
        # if self.update_count % 240 == 90:
        #     self.move_joints(5, None, 4, 4)
        # if self.update_count % 240 == 120:
        #     self.move_joints(-5, None, 4, 4)
        
        self.updating = False
        
    def move_shutter_by_gesture(self, hand_sign_id):
        '''
        Uses the most recent inputs (images, depths, and tracking)
        to calculate final joint positions for Shutter. These joint
        positions are then passed to move_joints.

        hand sign ids
        --------------
        no gesture: None
        peace sign: 0
        hand open: 1
        hand close: 2
        thumbs up: 3
        point up: 4
        point down: 5
        point left: 6
        point right: 7
        '''

        speed_boost = min(4, self.gesture_counter / 30)
        if self.gesture_counter >= 120:
            speed_boost = 4 + min(4, (self.gesture_counter - 120) / 15)

        if hand_sign_id == self.gesture_id:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 0
            self.gesture_id = hand_sign_id 

        if hand_sign_id == -1:
            if self.update_count % GESTURE_RATE == 0:
                print("NONE {}".format(self.gesture_counter))
            if self.gesture_counter >= 600:
                self.move_joints(0, -10, -10, -10, update_ideal = False)
        elif hand_sign_id == 0:
            if self.update_count % GESTURE_RATE == 0:
                print("PEACE {}".format(self.gesture_counter))
        elif hand_sign_id == 1:
            if self.update_count % GESTURE_RATE == 0:
                print("OPEN {}".format(self.gesture_counter))
            if self.joint_states is None:
                return
            if self.gesture_counter == 20:
                new_j1 = None
                new_j3 = None
                joint1 = None
                joint3 = None
                cur_joints = self.get_joints_list()

                if cur_joints is not None and self.ka_track is not None and len(self.ka_track.markers) > 0:
                    # compute the required motion to make the robot look towards the target
                    joint3 = cur_joints[2]
                    joint1 = cur_joints[0]
                    marker_msgs = filter_markers_by_type(self.ka_track.markers, [K4ABT_JOINT_HEAD])
                    marker_msg = None
                    if len(marker_msgs) == 1:
                        marker_msg = marker_msgs[0]
                        joint_angles = self.compute_joints_position(marker_msg, joint1, joint3)
                        if joint_angles is not None:
                            new_j1, new_j3 = joint_angles
                        self.move_joints(new_j1, None, new_j3, None, update_ideal=True)
        elif hand_sign_id == 2:
            if self.update_count % GESTURE_RATE == 0:
                print("CLOSE {}".format(self.gesture_counter))
        elif hand_sign_id == 3:
            if self.update_count % GESTURE_RATE == 0:
                print("THUMBS UP {}".format(self.gesture_counter))
            if self.gesture_counter == 20 and self.rs_image is not None:
                self.photo_image_pub.publish(self.rs_image)
        elif hand_sign_id == 4:
            if self.update_count % GESTURE_RATE == 0:
                print("POINT UP {}".format(self.gesture_counter))
            self.increment_joints(0, 0, 0.002 * speed_boost, 0)
        elif hand_sign_id == 5:
            if self.update_count % GESTURE_RATE == 0:
                print("POINT DOWN {}".format(self.gesture_counter))
            # the convenience limit is to make it easy for shutter to look at the user
            convenience_limit = self.moving_joints['joint_3'].limit.lower
            if self.ideal_joints[2] > convenience_limit - 0.002 * speed_boost + 0.2:
                self.increment_joints(0, 0, -0.002 * speed_boost, 0)
        elif hand_sign_id == 6:
            if self.update_count % GESTURE_RATE == 0:
                print("POINT LEFT {}".format(self.gesture_counter))
            self.increment_joints(-0.002 * speed_boost, 0, 0, 0)
        elif hand_sign_id == 7:
            if self.update_count % GESTURE_RATE == 0:
                print("POINT RIGHT {}".format(self.gesture_counter))
            self.increment_joints(0.002 * speed_boost, 0, 0, 0)
    
    def display_input_info(self):
        '''
        Displays desired info from each of the inputs.
        '''
        print("") # ends the sequence of dots from waiting
        print("&&&&&&&&&&&&&&&&")
        print("BEGINNING UPDATE")
        print("&&&&&&&&&&&&&&&&")
        print("Update #{}".format(self.update_count + 1))

        if self.rs_image is not None:
            print("rs_image:")
            rospy.loginfo("  shape: {}".format(self.imgmsg_to_cv2(self.rs_image).shape))

        # if self.rs_depth is not None:
        #     print("rs_depth:")
        #     rospy.loginfo("  shape: {}".format(self.rs_depth.shape))

        if self.ka_image is not None:
            print("ka_image:")
            rospy.loginfo("  shape: {}".format(self.ka_image.shape))

        # if self.ka_depth is not None:
        #     print("ka_depth:")
        #     rospy.loginfo("  shape: {}".format(self.ka_depth.shape))

        if self.ka_track is not None:
            print("ka_track:")
            rospy.loginfo("  marker count: {}".format(len(self.ka_track.markers)))
        
        if self.joint_states is not None:
            print("joint_states:")
            rospy.loginfo(self.joint_states)
        
        print("&&&&&&&&&&&&&&&&")
        print("COMPLETED UPDATE")
        print("&&&&&&&&&&&&&&&&")

    def get_joints_list(self):
        return self.to_joints_list(self.joint_states.position)
    
    def move_joints(self, j1 = None, j2 = None, j3 = None, j4 = None, update_ideal = False):
        '''
        Moves shutter's joints. More flexible
        :param float_list: a list of 4 floats
        '''
        if self.joint_states is None:
            return
        pos = self.get_joints_list()
        
        if j1 is None:
            j1 = pos[0]
        if j2 is None:
            j2 = pos[1]
        if j3 is None:
            j3 = pos[2]
        if j4 is None:
            j4 = pos[3]
        joints_list = self.to_joints_list([j1, j2, j3, j4])
        if update_ideal:
            self.ideal_joints = joints_list
        self.move_joints_raw(joints_list)

    def increment_joints(self, d1 = 0, d2 = 0, d3 = 0, d4 = 0):
        '''
        Moves shutter's joints in terms of displacement from their current position. 
        :param float_list: a list of 4 floats
        '''
        if self.joint_states is None:
            return
        pos = self.get_joints_list()
        if self.ideal_joints is not None:
            pos = self.ideal_joints
        joints_list = self.to_joints_list([d1 + pos[0], d2 + pos[1], d3 + pos[2], d4 + pos[3]])
        self.ideal_joints = joints_list
        self.move_joints_raw(joints_list)
    
    def to_joints_list(self, cand_list):
        '''
        Protects shutter's movements.
        :param cand_list: expected to be a list or tuple of 4 floats
        '''
        if not isinstance(cand_list, list) and not isinstance(cand_list, tuple):
            raise Exception("to_joints_list did not receive a list!")
        if len(cand_list) != 4:
            raise Exception("to_joints_list did not receive 4 joints!")
        
        sm_joints = self.moving_joints
        return [
            clamp_joint_angle(sm_joints['joint_1'].limit, cand_list[0]),
            clamp_joint_angle(sm_joints['joint_2'].limit, cand_list[1]),
            clamp_joint_angle(sm_joints['joint_3'].limit, cand_list[2]),
            clamp_joint_angle(sm_joints['joint_4'].limit, cand_list[3])
        ]

    def move_joints_raw(self, joints_list):
        '''
        Moves shutter's joints. There are NO SAFETY CHECKS here.
        :param joints_list: a list of 4 floats
        '''
        msg = Float64MultiArray()
        msg.data = joints_list
        self.joint_pub.publish(msg)

    def cleanup(self):
        """
        Do anything that needs to be done before closing.
        """
        self.move_joints(0, -10, -10, -10, update_ideal = False)
        print("Closing after {} updates!".format(self.update_count))
        
        
if __name__ == '__main__':
    try:
        track_hands = HandTrackingNode()
    except rospy.ROSInterruptException:
        pass

    sys.exit(0)
