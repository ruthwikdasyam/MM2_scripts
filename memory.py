#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Int32MultiArray
from nav_msgs.msg import Odometry
import time
import json
from datetime import datetime
from language import LanguageModels

class MemoryNode:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('memory_node', anonymous=True)

        rospy.Subscriber('/subtask', String, self.subtask_callback)
        rospy.Subscriber('/arm_pos', Int32MultiArray, self.armpos_callback)
        rospy.Subscriber('/user_query', String, self.user_query_callback)
        rospy.Subscriber('/response_sequence', String, self.response_sequence_callback)
        rospy.Subscriber('/response_reason', String, self.response_reason_callback)
        rospy.Subscriber('/task_status', String, self.task_status_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.subtask_name = "--"
        self.arm_pos = ['--', '--', '--', '--', '--', '--', '--']
        self.user_query = "--"
        self.response_sequence = "--"
        self.response_reason = "--"
        self.task_status = "--"
        self.odom_entry = ['--', '--', '--', '--', '--', '--', '--']
        self.loc_options = ["ruthwik", "zahir", "amisha", "kasra", "home"]
        self.arm_options = ["pickup", "dropoff"]

        self.llm = LanguageModels(loc_options=self.loc_options, arm_options=self.arm_options)
        

    # Callback functions
    def subtask_callback(self, data):
        self.subtask_name = data.data

    def armpos_callback(self, msg):
        self.arm_pos = msg.data
    
    def user_query_callback(self, data):
        self.user_query = data.data

    def response_sequence_callback(self, data):
        self.response_sequence = data.data
    
    def response_reason_callback(self, data):
        self.response_reason = data.data
    
    def task_status_callback(self, data):
        self.task_status = data.data
    
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
        log = {}
        # Timestamp
        log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Robot Status and Position
        log["robot"] = {
            "status": {
                "base_status": "Active" if self.subtask_name in self.loc_options else "Rest",
                "arm_status": "Active" if self.subtask_name in self.arm_options else "Rest"
            },
            "position": {
                "base_position": self.odom_entry,  #  list [x, y, z]
                "arm_position": self.arm_pos
            }}
        # LLM Logs
        log["llm"] = {
            "user_query": self.user_query,
            "response": self.response_sequence,
            "reasoning": self.response_reason
        }
        # Camera Observation
        try:
            log["camera_observation"] = self.llm.get_vlm_feedback(task="caption")
        except Exception as e:
            log["camera_observation"] = f"Error: Could not capture image. ({str(e)})"
        # Task Progress
        log["task_progress"] = {
            "task_name": self.subtask_name,
            "task_status": self.task_status
        }
        # Return the log as a JSON string
        return json.dumps(log, indent=4)



if __name__ == '__main__':
    try:
        mem_node = MemoryNode()

        while True:
            log = mem_node.get_log()
            print(log)
            # time.sleep(1)
            break
    except rospy.ROSInterruptException:
        pass