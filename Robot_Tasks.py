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


"""
Functions to include

1. Navigate to person
2. Navigate to Point
3. Navigate to object
4. Find object
4. Manipulate
   - pickup
   - dropoff
3. Ask user

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
        self.arm_options = ["pickup", "dropoff"]
        
        # Instantiating
        # self.mygello = GELLOcontroller("doodle", torque_start=True)
        self.llm = LanguageModels(loc_options=self.loc_options, arm_options=self.arm_options)

        # publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10) # publishes goal point
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)         # cancels all goals- WAIT feature

        self.task_status_pub = rospy.Publisher('/task_status', String, queue_size=10)
        self.subtask_pub = rospy.Publisher('/subtask', String, queue_size=10)                # publishes current task name
        self.parameter_pub = rospy.Publisher('/parameter', String, queue_size=10)                # publishes current task name
        self.askuser_pub = rospy.Publisher('/askuser', String, queue_size=10)  # to ask user for input


        # subscribers
        # rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self._rs_callback)
        rospy.Subscriber('/highlevel_response', String, self.sequence_callback)           # reading robot status
        rospy.Subscriber('/user_query', String, self.input_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)           # reading robot status

        # Initialize variables
        self.sequence = ""
        self.possible_tasks = ["go_to_person", "go_to_point", "approach_object", "get_image_caption", "set_arm_position", "ask_user"]
        self.vlm_for_gripper = 0

        self.active_server = "" #["movebase","arm"]


    # PRIVATE METHODS
    def sequence_callback(self, msg):
        self.sequence = msg
        # print(self.sequence)

    def input_callback(self,msg):
        self.user_input = msg
        if self.user_input == "wait":
            self.wait()
        self.sequence=""

    def status_callback(self, msg):
        self.tb_status= msg.status_list[-1].status

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
        

    # PUBLIC METHODS
    def go_to_person(self, name: str):
        '''
        Input: place name {str},
        Output: Robot moves
        '''
        self.active_server = "movebase"
        selected_pose = self.pose_dict[name]
        goal = PoseStamped()
        goal.header = selected_pose.header
        goal.pose = selected_pose.pose.pose
        for i in range(2):
            self.goal_pub.publish(goal)   # publishing navigation goal
            time.sleep(1)


    def go_to_point(self, coordinate: tuple):
        '''
        Input: coordinate {tuple} - (x, y, z, x, y, z, w)
        Ouput: Robot moves
        '''
        self.active_server = "movebase"
        # check for format and limits
        assert len(coordinate) == 7, "Coordinate should be a tuple of length 6"
        assert all(isinstance(i, (int, float)) for i in coordinate), "All elements should be int or float"

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


    def approach_object(self, object_name: str):
        '''
        Input: object_name {str}
        Ouput: Robot moves with visual navigation
        '''
        self.active_server = "movebase"

    def get_image_caption(self):
        '''
        Output: Caption for image {str}
        '''
        caption = self.llm.get_vlm_feedback(task="caption", rs_image=self.image)
        return caption

    def set_arm_position(self, state: str):
        '''
        Input: state of manipulator {str}
        Output: Manipulator moves
        '''
        self.active_server = "arm"
        print(f"Arm is {state}")
        if state == "pickup":
            self.mygello.pickup()
            while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                if self.llm.get_vlm_feedback(state)==1:
                    break
                time.sleep(0.5)
            self.mygello.pickup_complete()

        elif state == "dropoff":
            self.mygello.dropoff()
            while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                if self.llm.get_vlm_feedback(state)==1:
                    break
                time.sleep(0.5)
            self.mygello.dropoff_complete()


    def ask_user(self, data:str):
        '''
        Input: what to ask user {str}
        Output: Asks user
        '''
        
        # this ask user thing should go to main input
        user_response = input(f"Hey user: {data}")
        # publish to use input, and break this code
        self.askuser_pub.publish(user_response)
        self.wait()


    def wait(self):
        '''
        Output: Stops everything and waits
        '''
        self.cancel_pub.publish(GoalID())



    


if __name__ == "__main__":
    coco = RobotTasks()

    # verify the sequence
    # assert coco.sequence != "", "No sequence received"
    print(f"Sequence: {coco.sequence}")

    while not rospy.is_shutdown():
        if coco.sequence != "":
            # Parse the JSON string
            seq = json.loads(coco.sequence.data)
            # Loop through the list of steps inside the "steps" key
            # for step in seq["steps"]:
            step = seq["steps"][0]
            print(step)
            if step["task"] in coco.possible_tasks:
                getattr(coco, step["task"])(step["parameter"])  # Call the method dynamically
                
                coco.subtask_pub.publish(step["task"])
                coco.parameter_pub.publish(step["parameter"])

                if coco.active_server == "movebase":
                    if coco.tb_status == 3:
                        coco.task_status_pub.publish("completed")
                    else:
                        coco.task_status_pub.publish("running")
                # elif coco.active_server == "arm":
                #     if coco.arm_status == 3:
                #         coco.task_status_pub.publish("completed")
                #     else:
                #         coco.task_status_pub.publish("running")
    
            else:
                print(f"Unknown task: {step['task']}")
        time.sleep(1)
