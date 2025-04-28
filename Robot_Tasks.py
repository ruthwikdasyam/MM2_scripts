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
import ast

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

"""
Functions to include


"navigate_to_person":[f"one person_name from {self.loc_options} only"],
"navigate_to_position":["x","y","z","w1","w2","w3","w4"],
"navigate_to_object":["object_name"],
"manipulate":[f"one function_name from {self.arm_options} only"],
"get_image_caption":["prompt on what you want to know"],
"ask_user":["question"],
"wait":[],

"""


class RobotTasks:
    def __init__(self):
        rospy.init_node("robot_tasks", anonymous=True)

        # Accessing saved locations
        self.pose_dict = {}        
        location_map = json.load(open("/home/nvidia/catkin_ws/src/nav_assistant/jsons/location_pose_map.json"))        
        for key, fl in location_map.items():            
            self.pose_dict[key] = self.read_pose_from_file(f"/home/nvidia/catkin_ws/src/nav_assistant/poses/{fl}.txt")        
        self.loc_options = ', '.join(list(location_map.keys()))
        self.arm_options = ["start_pickup","complete_pickup","start_dropoff","complete_dropoff"]
        
        # Instantiating
        self.mygello = GELLOcontroller("doodle", torque_start=True)
        self.llm = LanguageModels()

        # publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10) # publishes goal point
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)         # cancels all goals- WAIT feature

        self.task_status_pub = rospy.Publisher('/task_status', String, queue_size=10)
        self.subtask_pub = rospy.Publisher('/subtask', String, queue_size=10)                # publishes current task name
        self.parameter_pub = rospy.Publisher('/parameter', String, queue_size=10)                # publishes current task name
        self.askuser_pub = rospy.Publisher('/askuser', String, queue_size=10)  # to ask user for input
        self.user_input_pub = rospy.Publisher('/user_input', String, queue_size=10)
        self.task_info_pub = rospy.Publisher('/task_info', String, queue_size=10)


        # subscribers
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.rs_callback)
        rospy.Subscriber('/highlevel_response', String, self.sequence_callback)           # reading robot status
        rospy.Subscriber('/user_input', String, self.input_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)           # reading robot status
        rospy.Subscriber('/task_status', String, self.task_status_callback)

        # Initialize variables
        self.sequence = ""
        self.possible_tasks = ["navigate_to_person", "navigate_to_position", "navigate_to_object", "get_image_caption", "manipulate", "ask_user", "wait"]
        self.vlm_for_gripper = 0

        self.active_server = "" #["movebase","arm"]


    # PRIVATE METHODS
    def sequence_callback(self, msg):
        self.sequence = msg
        # print(self.sequence)
        self.task_status_pub.publish(" ")

    def input_callback(self,msg):
        self.user_input = msg
        if self.user_input == "wait":
            self.wait()
        self.sequence=""

    def status_callback(self, msg):
        self.tb_status= msg.status_list[-1].status
        self.tb_feedback = msg.status_list[-1].text

    def task_status_callback(self, data):
        self.task_status = data.data

    def rs_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
            orientation=Quaternion(orientation['x'], orientation['y'], orientation['z'], orientation['w'])
            )
        pose_msg.pose.covariance = covariance
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rospy.Time.now()
        return pose_msg
    

    def check_status(self):
        '''
        Output: status of robot {int}
        '''
        pass    

        

    # PUBLIC METHODS
    def navigate_to_person(self, name: str):
        '''
        Input: place name {str},
        Output: Robot moves
        '''
        self.active_server = "movebase"
        selected_pose = self.pose_dict[name.lower()]
        goal = PoseStamped()
        goal.header = selected_pose.header
        goal.pose = selected_pose.pose.pose
        for i in range(2):
            self.goal_pub.publish(goal)   # publishing navigation goal
            time.sleep(1)


    def navigate_to_position(self, coordinate):
        '''
        Input: coordinate {tuple} - (x, y, z, x, y, z, w)
        Ouput: Robot moves
        '''
        coordinate = tuple(ast.literal_eval(coordinate))
        
        self.active_server = "movebase"
        # print(f"coordinate {coordinate}. {type(coordinate)}")
        # check for format and limits
        assert len(coordinate) == 7, "Coordinate should be a tuple of length 7"
        # assert all(isinstance(i, (int, float)) for i in coordinate), "All elements should be int or float"

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose = Pose(
            position=Point(coordinate[0], coordinate[1], coordinate[2]),
            orientation=Quaternion(coordinate[3], coordinate[4], coordinate[5], coordinate[6])
            )
        for i in range(2):
            self.goal_pub.publish(goal)   # publishing navigation goal
            time.sleep(1)


    def navigate_to_object(self, object_name: str):
        self.active_server = "movebase"

    def get_image_caption(self, data):

        # self.task_status_pub.publish("running")
        response = self.llm.get_vlm_feedback(task="caption_2", rs_image=self.image, question=data)

        self.task_info_pub.publish(response)
        self.task_status_pub.publish("completed")

    def manipulate(self, state: str):
        '''
        Input: state of manipulator {str}
        Output: Manipulator moves
        '''
        self.active_server = "arm"
        print(f"Arm {state}")

        if not self.mygello.is_near_target(state):
            self.task_status_pub.publish("running")

            if state == "start_pickup":
                self.mygello.pickup()
                # while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                #     if self.llm.get_vlm_feedback(state)==1:
                #         break
                #     time.sleep(0.5)
                # self.mygello.pickup_complete()
            elif state == "complete_pickup":
                self.mygello.pickup_complete()
            elif state == "start_dropoff":
                self.mygello.dropoff()
                # while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                #     if self.llm.get_vlm_feedback(state)==1:
                #         break
                #     time.sleep(0.5)
                # self.mygello.dropoff_complete()
            elif state == "complete_dropoff":
                self.mygello.dropoff_complete()

        self.task_status_pub.publish("completed")
        # self.sequence=" "
        # time.sleep(4)

    def ask_user(self, data:str):
        '''
        Input: what to ask user {str}
        Output: Asks user
        '''
        # this ask user thing should go to main input
        # publish to use input, and break this code
        self.task_status_pub.publish("running")
        self.askuser_pub.publish(data)
        self.wait()
        self.sequence = ""


    def wait(self, dummy=" "):
        '''
        Output: Stops everything and waits
        '''
        self.cancel_pub.publish(GoalID())
        self.sequence = ""


    


if __name__ == "__main__":
    coco = RobotTasks()

    # verify the sequence
    # assert coco.sequence != "", "No sequence received"
    print(f"Sequence: {coco.sequence}")

    while not rospy.is_shutdown():
        if coco.sequence != "":
            coco.active_server = ""
            # Parse the JSON string
            seq = json.loads(coco.sequence.data)
            # Loop through the list of steps inside the "steps" key
            # for step in seq["steps"]:
            step = seq["steps"][0]

            # try:
            #     if current_task == step["task"] and current_param == step["parameter"] and coco.task_status == "completed":
            #         step = seq["steps"][1]
            # except:
            #     pass
            # coco.task_info_pub.publish(" ")
            print(step)
            if step["task"] in coco.possible_tasks:
                coco.subtask_pub.publish(step["task"])
                coco.parameter_pub.publish(step["parameter"])
                
                getattr(coco, step["task"])(step["parameter"])  # Call the method dynamically
                
                if coco.active_server == "movebase":
                    coco.task_info_pub.publish(coco.tb_feedback)
                    if coco.tb_status == 3:
                        coco.task_status_pub.publish("completed")
                    else:
                        coco.task_status_pub.publish("running")

                elif coco.active_server == "arm":
                    coco.task_info_pub.publish(" ")
                #     if coco.arm_status == 3:
                #         coco.task_status_pub.publish("completed")
                #     else:
                #         coco.task_status_pub.publish("running")

                # current_task = step["task"]
                # current_param = step["parameter"]

                

            else:
                print(f"Unknown task: {step['task']}")
                coco.user_input_pub.publish(f"Unknown task: {step['task']}")
        time.sleep(1)

    # coco.wait()
    # coco.navigate_to_person('zahir')
    # coco.navigate_to_position('-2.8977617696865288, 6.7348915347955565, 0.0, 0.0, 0.0, 0.5825486289046017, 0.8127958507284401')
    # coco.navigate_to_position('6.836892185164943, 5.983559917708442, 0.0, 0.0, 0.0, 0.12264594311087945, 0.9924504887592343')
    # coco.navigate_to_position('-2.8, 6.3, 0.0, 0.0, 0.0, 0.5, 0.8')

