#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Int32MultiArray
from nav_msgs.msg import Odometry
import time
import json
from datetime import datetime
from language import LanguageModels
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped

class MemoryNode:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('memory_node', anonymous=True)

        rospy.Subscriber('/subtask', String, self.task_name_callback)
        rospy.Subscriber('/parameter', String, self.parameter_callback)
        rospy.Subscriber('/task_status', String, self.task_status_callback)

        rospy.Subscriber('/user_query', String, self.user_query_callback)
        rospy.Subscriber('/response_plan', String, self.response_plan_callback)
        rospy.Subscriber('/response_reason', String, self.response_reason_callback)
        rospy.Subscriber('/highlevel_response', String, self.sequence_callback)

        rospy.Subscriber('/armpos', Int32MultiArray, self.armpos_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)

        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.rs_callback)

        self.task_name = "--"
        self.parameter = '--'
        self.arm_pos = str(['--', '--', '--', '--', '--', '--', '--'])
        self.user_query = "--"
        self.response_plan = "--"
        self.response_reason = "--"
        self.task_status = "--"
        self.odom_entry = " "
        self.amcl_entry = " "
        self.loc_options = ["ruthwik", "zahir", "amisha", "kasra", "home"]
        self.arm_options = ["start_pickup","complete_pickup","start_dropoff","complete_dropoff"]
        self.image = None

        self.generate_captions = True

        self.llm = LanguageModels()
        

    # Callback functions
    def task_name_callback(self, data):
        self.task_name = data.data

    def parameter_callback(self, data):
        self.parameter = data.data

    def armpos_callback(self, msg):
        self.arm_pos = msg.data
    
    def user_query_callback(self, data):
        self.user_query = data.data

    def response_plan_callback(self, data):
        self.response_plan = data.data
    
    def response_reason_callback(self, data):
        self.response_reason = data.data
    
    def sequence_callback(self, data):
        self.sequence = data.data
        log = self.get_log(type="llm")
        print(json.dumps(log, indent=4))
        self.save_logs(log)
    
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
        self.odom_entry = str([x, y, z, ox, oy, oz, ow])
   
    def amcl_callback(self, msg):
        # Extract position (x, y, z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Extract orientation (quaternion: x, y, z, w)
        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w

        # Store all in a stringified array
        self.amcl_entry = str([x, y, z, ox, oy, oz, ow])

    def rs_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def get_log(self, type):

        log = {}
        if type == "status":
            # Timestamp
            log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log["type"] = "status"
            # Robot Status and Position
            log["robot"] = {
                # "status": {
                #     "base_status": "Active" if self.task_name in self.loc_options else "Rest",
                #     "arm_status": "Active" if self.task_name in self.arm_options else "Rest"
                # },
                "position": {
                    "base_position": self.amcl_entry,  #  list [x, y, z]
                    "arm_position": self.arm_pos
                }}
            
            # log["llm"] = {
            #     "user_query": self.user_query,
            #     "response": self.response_plan,
            #     "reasoning": self.response_reason
            # }
            # Camera Observation
            # try:
            if self.image is not None and self.generate_captions is True:
                log["camera_observation"] = self.llm.get_vlm_feedback(task="caption", rs_image=self.image)
            else:
                log["camera_observation"] = " "
            # except Exception as e:
            #     log["camera_observation"] = f"Error: Could not capture image. ({str(e)})"
            # Task Progress
            log["task_progress"] = {
                "task_name": self.task_name,
                "parameter": self.parameter,
                "task_status": self.task_status
            }
            # Return the log as a JSON string
            return log
        
        elif type == "llm":
            # Timestamp
            log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log["type"] = "llm"
            # LLM Logs
            log["llm"] = {
                "user_query": self.user_query,
                "response": self.response_plan,
                "reasoning": self.response_reason,
                "sequence": self.sequence
            }
            return log


    def save_logs(self, log):
        log_file = "memory_files/robot_logs.jsonl"  # JSONL (JSON Lines) format for continuous logging
        try:
            json_line = json.dumps(log) 
            with open(log_file, "a") as file:
                file.write(json_line + "\n")
        except KeyboardInterrupt:
            print("Logging stopped by user.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    # try:
        mem_node = MemoryNode()

        while not rospy.is_shutdown():
            print("----------------------------------------------------------------------")
            log = mem_node.get_log(type="status")
            print(json.dumps(log, indent=4))
            mem_node.save_logs(log)
            # if mem_node.image is None:
            time.sleep(2)
            # break
    # except rospy.ROSInterruptException:
    #     break
    