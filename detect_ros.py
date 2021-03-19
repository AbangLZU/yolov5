import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


import os
import tempfile
import threading
from six.moves import urllib

import PIL
import numpy as np

from cv_bridge_cus import CvBridge
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


import yaml
from sensor_msgs.msg import CameraInfo
from utils import citiscapes_labels as label_util


def yaml_to_CameraInfo(yaml_fname):
    """
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

class YoloNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 1)
        self._visualize = rospy.get_param('~visualize', True)
        self._model_dir = rospy.get_param('~model_dir', 'models')
        self._intrinsics_file = rospy.get_param('~camera_intrinsics_file', '')
        print('camera intrinstics yaml file path: ', self._intrinsics_file)

        if not self._intrinsics_file == '':
            self._camera_info = yaml_to_CameraInfo(self._intrinsics_file)
            print("load yaml file success, camera width: ", self._camera_info.width)

        rgb_input = rospy.get_param('~rgb_input', '/camera/rgb/image_color')

        rospy.Subscriber(rgb_input, Image, self._image_callback, queue_size=1)

        self.label_pub = rospy.Publisher('~yolo_detection_label', Image, queue_size=1)
        self.vis_pub = rospy.Publisher('~yolo_detection_viz', Image, queue_size=1)
        self.cam_info_pub = rospy.Publisher("~camera_info", CameraInfo, queue_size=1)

        weight_file = 'yolov5m.pt'
        self._model_path = os.path.join(self._model_dir, weight_file)

        # load the model
        rospy.loginfo('start load model.....')

        print('Model loaded successfully!')
        self._load_neural_net = True

        with torch.no_grad():
            detect()



    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():

            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if not self._intrinsics_file == '':
                self.cam_info_pub.publish(self._camera_info)


            if msg is not None:
                rgb_image = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                # Run detection.
                seg_map = self.detect(rgb_image)
                # print("finish inference...")

                if self._visualize:
                    # Overlay segmentation on RGB image.
                    image = self.visualize(rgb_image, seg_map)
                    label_color_msg = self._cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                    label_color_msg.header = msg.header
                    self.vis_pub.publish(label_color_msg)
                    # rospy.loginfo('publish vis image')
            rate.sleep()

    def detect(self, rgb_image):
        rgb_image = PIL.Image.fromarray(rgb_image)

        resized_im, seg_map = self._model.run(rgb_image)
        seg_map = cv2.resize(seg_map.astype(np.float32), rgb_image.size,
                             interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        return seg_map

    def visualize(self, rgb_image, seg_map, alpha=0.4):
        pass

    def _image_callback(self, msg):
        # print("Got an image.")

        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()
    


def main():
    rospy.init_node('yolo_ros_node')

    node = YoloNode()
    node.run()


if __name__ == '__main__':
    main()