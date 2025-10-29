import warnings
import sys
import rospy
import numpy as np
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray
from urdf_parser_py.urdf import URDF
import cv2
from cv_bridge import CvBridge, CvBridgeError
import app
from scipy.optimize import least_squares
import tf2_geometry_msgs
import tf.transformations as tft

# equivalent of an enum for joint tracking
K4ABT_JOINT_PELVIS=0
K4ABT_JOINT_SPINE_NAVEL=1
K4ABT_JOINT_SPINE_CHEST=2
K4ABT_JOINT_NECK=3
K4ABT_JOINT_CLAVICLE_LEFT=4
K4ABT_JOINT_SHOULDER_LEFT=5
K4ABT_JOINT_ELBOW_LEFT=6
K4ABT_JOINT_WRIST_LEFT=7
K4ABT_JOINT_HAND_LEFT=8
K4ABT_JOINT_HANDTIP_LEFT=9
K4ABT_JOINT_THUMB_LEFT=10
K4ABT_JOINT_CLAVICLE_RIGHT=11
K4ABT_JOINT_SHOULDER_RIGHT=12
K4ABT_JOINT_ELBOW_RIGHT=13
K4ABT_JOINT_WRIST_RIGHT=14
K4ABT_JOINT_HAND_RIGHT=15
K4ABT_JOINT_HANDTIP_RIGHT=16
K4ABT_JOINT_THUMB_RIGHT=17
K4ABT_JOINT_HIP_LEFT=18
K4ABT_JOINT_KNEE_LEFT=19
K4ABT_JOINT_ANKLE_LEFT=20
K4ABT_JOINT_FOOT_LEFT=21
K4ABT_JOINT_HIP_RIGHT=22
K4ABT_JOINT_KNEE_RIGHT=23
K4ABT_JOINT_ANKLE_RIGHT=24
K4ABT_JOINT_FOOT_RIGHT=25
K4ABT_JOINT_HEAD=26
K4ABT_JOINT_NOSE=27
K4ABT_JOINT_EYE_LEFT=28
K4ABT_JOINT_EAR_LEFT=29
K4ABT_JOINT_EYE_RIGHT=30
K4ABT_JOINT_EAR_RIGHT=31
K4ABT_JOINT_COUNT=32

# the indices that we care about
relevant_joint_indices = (
    K4ABT_JOINT_WRIST_LEFT,
    K4ABT_JOINT_HAND_LEFT,
    K4ABT_JOINT_HANDTIP_LEFT,
    K4ABT_JOINT_THUMB_LEFT,
    K4ABT_JOINT_WRIST_RIGHT,
    K4ABT_JOINT_HAND_RIGHT,
    K4ABT_JOINT_HANDTIP_RIGHT,
    K4ABT_JOINT_THUMB_RIGHT
)

RAISE_WARN = False # should warnings stop the robot from running?

def transform_msg_to_T(trans):
    """
    Convert TransformStamped message to 4x4 transformation matrix
    :param trans: TransformStamped message
    :return:
    """
    # extract relevant information from transform
    q = [trans.transform.rotation.x,
         trans.transform.rotation.y,
         trans.transform.rotation.z,
         trans.transform.rotation.w]
    t = [trans.transform.translation.x,
         trans.transform.translation.y,
         trans.transform.translation.z]
    # convert to matrices
    Rq = tft.quaternion_matrix(q)
    Tt = tft.translation_matrix(t)
    return np.dot(Tt, Rq)

def make_joint_rotation(angle, rotation_axis='x'):
    """
    Make rotation matrix for joint (assumes that joint angle is zero)
    :param angle: joint angle
    :param rotation_axis: rotation axis as string or vector
    :return: rotation matrix
    """
    # set axis vector if input is string
    if not isinstance(rotation_axis,list):
        assert rotation_axis in ['x', 'y', 'z'], "Invalid rotation axis '{}'".format(rotation_axis)
        if rotation_axis == 'x':
            axis = (1.0, 0.0, 0.0)
        elif rotation_axis == 'y':
            axis = (0.0, 1.0, 0.0)
        else:
            axis = (0.0, 0.0, 1.0)
    else:
        axis = rotation_axis
    # make rotation matrix
    R = tft.rotation_matrix(angle, axis)
    return R

def target_in_camera_frame(angles, target_pose, rotation_axis1, rotation_axis2, T1, T2):
    """
    Transform target to camera frame
    :param angles: joint angles
    :param target_pose: target pose
    :param rotation_axis1: str representation for the rotation axis of joint1
    :param rotation_axis2: str representation for the rotation axis of joint3
    :param T1: transform - base_link to biceps
    :param T2: transform - biceps to camera_link
    :return: target in camera_link, target in base_link
    """

    # make transform for joint 1
    R1 = make_joint_rotation(angles[0], rotation_axis=rotation_axis1)

    # make transform for joint 3
    R2 = make_joint_rotation(angles[1], rotation_axis=rotation_axis2)

    # transform target to camera_link...
    p = np.array([[target_pose[0], target_pose[1], target_pose[2], 1.0]]).transpose()

    # target in base_link
    p1 = np.dot(np.dot(T1, R1), p)

    # target in camera_link
    result = np.dot(np.dot(T2, R2), p1)

    return result[0:2].flatten()

def warn_msg(msg):
    '''
    Emits a warning or an exception, depending on RAISE_WARN.
    :param msg: a string
    '''
    if RAISE_WARN:
        raise Exception(msg)
    else:
        warnings.warn(msg)


def filter_markers_by_type(markers, joint_indices = relevant_joint_indices):
    '''
    Emits a warning or an exception, depending on RAISE_WARN.
    :param markers: a list / tuple of Marker objects
    :param types: a list / tuple of numbers from the gesture enum
    '''
    result = []
    for marker in markers:
        # for the kinect, id % 100 gives the joint_index 
        if marker.id % 100 in joint_indices:
            result.append(marker)
    return result


def clamp_joint_angle(joint_limit, angle):
    '''
    Bounds a joint angle based on known limits.
    :param joint_limit: a urdf_parser_py.urdf.JointLimit object
    :param angle: a number that can be cast to a float
    '''
    lower = joint_limit.lower
    upper = joint_limit.upper
    if lower > upper:
        raise Exception("Joint has invalid limits!")
    if angle > upper:
        warn_msg("Received an angle above the joint limit!")
        angle = upper
    if angle < lower:
        warn_msg("Received an angle below the joint limit!")
        angle = lower
    return float(angle)
