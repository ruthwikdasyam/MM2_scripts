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
from std_msgs.msg import String, Int32MultiArray, Float64
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


class HighLevelInference:
    def __init__(self):
        rospy.init_node("random_goal_setter", anonymous=True)

        # Load location map
        self.pose_dict = {}        
        location_map = json.load(open("/home/nvidia/catkin_ws/src/nav_assistant/jsons/location_pose_map.json"))        
        for key, fl in location_map.items():            
            self.pose_dict[key] = self.read_pose_from_file(f"/home/nvidia/catkin_ws/src/nav_assistant/poses/{fl}.txt")        
        self.loc_options = ', '.join(list(location_map.keys()))
        self.arm_options = ["pickup", "dropoff"]

        # initializing language model
        self.llm = LanguageModels()
        # self.mygello = GELLOcontroller("doodle", torque_start=True)

        # publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10) # publishes goal point
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)         # cancels all goals- WAIT feature

        # memory
        self.subtask_name = rospy.Publisher('/subtask', String, queue_size=10)                # publishes current task name
        self.arm_pos = rospy.Publisher('/armpos', Int32MultiArray, queue_size=10)             # publishes current pos of manip
        # self.user_query = rospy.Publisher('/user_input', String, queue_size=10)
        self.response_plan = rospy.Publisher('/response_plan', String, queue_size=10)
        self.response_reason = rospy.Publisher('/response_reason', String, queue_size=10)
        self.sequence = rospy.Publisher('/highlevel_response', String, queue_size=10)
        self.task_status = rospy.Publisher('/task_status', String, queue_size=10)
        
        # subscribers
        # rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)           # reading robot status
        rospy.Subscriber('/odom', Odometry, self.odom_callback)                                # reading from odom - current position
        rospy.Subscriber('/user_input', String, self.user_query_callback)
        rospy.Subscriber('/askuser', String, self.askuser_callback)
        rospy.Subscriber('/time_newtask', Float64, self.time_newtask_callback)

        # initialize variables
        self.tb_status = 0
        self.breaking = False
        self.current_pose = 0
        self.run_now = 0
        self.user_query_sub = ""
        # params
        self.vlm_for_gripper = False  # Gripper using vlm to open and close during pickup and drop off
        self.run_high_level = 1
        self.task_start_time = 0

    def time_newtask_callback(self, data):
        self.task_start_time = data.data

    def odom_callback(self, msg):
        self.current_pose = msg

    # def status_callback(self, msg):
    #     self.tb_status= msg.status_list[-1].status

    def user_query_callback(self, data):
        self.user_query_sub = data.data
        self.run_high_level = 1
        self.run_now = 1

    def askuser_callback(self, data):
        self.askuser_sub = data.data
        self.run_high_level = 0

    def read_pose_from_file(self, filename):
        with open(filename, 'r') as file:
            parsed_data = yaml.safe_load(file)
        pose_data = parsed_data['pose']
        position = pose_data['position']
        orientation = pose_data['orientation']
        covariance = parsed_data['covariance']
        # Create a PoseWithCovarianceStamped message
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.pose.pose = Pose(
            position=Point(position['x'], position['y'], position['z']),
            orientation=Quaternion(orientation['x'], orientation['y'], orientation['z'], orientation['w']))
        pose_msg.pose.covariance = covariance
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rospy.Time.now()
        return pose_msg
    

    def run(self):

        # user input
        # query = input("Hello! How can i help :)")
        query = self.user_query_sub
        # Get Keywords
        keywords = self.llm.generate_keywords(query)
        print(keywords)
        # Filter Experiences
        self.llm.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
        self.llm.get_recent_20_experiences("memory_files/robot_logs.jsonl", "memory_files/recent_experiences.jsonl", newtask_time=self.task_start_time)
        # """
        # Step 1 Response
        step1_response = self.llm.get_response(user_query=query)
        # print(f"\n{step1_response.plan}")
        print(f"\nReason: \n{step1_response.reason}\n")
        # Step 2 Response
        self.llm.get_recent_20_experiences("memory_files/robot_logs.jsonl", "memory_files/recent_experiences.jsonl", newtask_time=self.task_start_time)
        step2_response = self.llm.get_response_sequence(plan=step1_response.plan, reason=step1_response.reason)
        data = json.loads(step2_response)
        output_lines = []
        for i, step in enumerate(data['steps'], start=1):
            print(f"{i}. Task: {step['task']} : {step['parameter']}")
            # output_lines.append(f"{i}. Task: {step['task']}, Parameter: {step['parameter']}")
        # print(output_lines)
        print("--------------------------------------------")
        # publishing data
        self.response_plan.publish(str(step1_response.plan))
        self.response_reason.publish(str(step1_response.reason))
        self.sequence.publish(step2_response)
        # """




if __name__=="__main__":
    hlic=HighLevelInference()
    hlic.llm.connection_check()
    # hlic.run()
    ch1 = time.time()
    while not rospy.is_shutdown():
        if hlic.run_high_level == 1:
            if hlic.run_now == 1 or time.time() - ch1 >= 5:
                ch1 = time.time()
                print("Thinking...")
                hlic.run()
                hlic.run_now = 0
        time.sleep(1)


  

