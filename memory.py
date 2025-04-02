#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Int32MultiArray
from nav_msgs.msg import Odometry
import time
from datetime import datetime
from language import LanguageModels

class MemoryNode:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('memory_node', anonymous=True)

        rospy.Subscriber('/subtask', String, self.subtask_callback)
        rospy.Subscriber('/arm_pos', Int32MultiArray, self.armpos_callback)
        rospy.Subscriber('/user_query', String)
        rospy.Subscriber('/response_sequence', String)
        rospy.Subscriber('/response_reason', String)


        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.subtask_name = " "
        self.loc_options = ["ruthwik", "zahir", "amisha", "kasra", "home"]
        self.arm_options = ["pickup", "dropoff"]

        self.llm = LanguageModels()
        


    # Callback functions
    def subtask_callback(self, data):
        self.subtask_name = data.data

    # def armpos_callback(self, )
    
    def odom_callback(self, msg):
        # Extract position (x, y, z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Extract orientation (quaternion: x, y, z, w)
        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w

        # Store all in an array
        self.odom_entry = [x, y, z, ox, oy, oz, ow]


    def get_log(self):
        log={}        # initializing a dictionary

        # Timestamp
        log["time"] = datetime.now.strftime("%Y-%m-%d %H:%M:%S")
        # Robot Status
        if self.subtask_name in self.loc_options:
            log["tb_status"] = "Active"
            log["arm_status"] = "Rest"
        elif self.subtask_name in self.arm_options:
            log["tb_status"] = "Rest"
            log["arm_status"] = "Active"
        else:
            log["tb_status"] = "Rest"
            log["arm_status"] = "Rest"

        log["tb_pos"] = "self.odom_entry"
        log["arm_pos"] = ""

        #LLM
        # log[]



if __name__ == '__main__':
    try:
        mem_node = MemoryNode()

        while True:
            print(f"-- {mem_node.task_name}")
            time.sleep(1)
    except rospy.ROSInterruptException:
        pass