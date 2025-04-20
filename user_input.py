import rospy
import yaml 
import time 
import json
from dataclasses import dataclass, field
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from mobilegello.gello_controller import GELLOcontroller
from language import LanguageModels
from actionlib_msgs.msg import GoalID
from std_msgs.msg import String, Int32MultiArray
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


class UserInputNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('user_input_node', anonymous=True)

        # Initialize variables


        # Initialize language model
        self.llm = LanguageModels(loc_options=self.loc_options, arm_options=self.arm_options)

        # Initialize publishers
        self.input_pub = rospy.Publisher('/user_query', String, queue_size=10)


    def publish_user_input(self):
        # Get user input
        input_text = input("Enter your command: ")
        # Publish the user input
        self.input_pub.publish(input_text)
        rospy.loginfo(f"Published user input: {input_text}")
        # Sleep for a bit to ensure the message is sent
        rospy.sleep(0.1)
        return
    

if __name__ == '__main__':
    try:
        user_input_node = UserInputNode()
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            user_input_node.publish_user_input()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

        