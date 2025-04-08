#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
# import os

class RealSenseImageSaver:
    def __init__(self):
        rospy.init_node('realsense_image_saver', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback)
        rospy.loginfo("Image saver initialized. Waiting for images...")

    def callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite("now_image_realsense.jpg", cv_image)
            rospy.loginfo(f"Saved image")
            
        except Exception as e:
            rospy.logerr(f"Failed to save image: {e}")

if __name__ == '__main__':
    camera_image = RealSenseImageSaver()
    rospy.spin()
