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
from std_msgs.msg import String



class RandomGoalSetter:
    def __init__(self):
        rospy.init_node("random_goal_setter", anonymous=True)

        # Load location map
        self.pose_dict = {}        
        location_map = json.load(open("/home/nvidia/catkin_ws/src/nav_assistant/jsons/location_pose_map.json"))        
        for key, fl in location_map.items():            
            self.pose_dict[key] = self.read_pose_from_file(f"/home/nvidia/catkin_ws/src/nav_assistant/poses/{fl}.txt")        
        self.loc_options = ', '.join(list(location_map.keys()))
        self.arm_options = ("pickup", "dropoff")

        # initializing language model
        self.llm = LanguageModels(loc_options=self.loc_options, arm_options=self.arm_options)
        self.mygello = GELLOcontroller("doodle", torque_start=True)

        self.tb_status = 0
        self.breaking = False
        self.current_pose = 0

        # move base goal to set a goal
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        # subscribing to odom to get current position
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        # publisher to movebase/cancel topic
        self.cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)

        rospy.loginfo("Monitoring robot status continuously...")
        self.continuously_monitor()

        # set parameters
        self.vlm_for_gripper = False  # Gripper using vlm to open and close during pickup and drop off



        # Memory logs
        self.subtask_name = rospy.Publisher('/subtask', String, queue_size=10)
        self.subtask_name.publish("RD")

    
    def odom_callback(self, msg):
        self.current_pose = msg
        # print(self.current_pose)
            
    def status_callback(self, msg):
        self.tb_status= msg.status_list[-1].status

    def continuously_monitor(self):
        # rospy.init_node('move_base_monitor')  # Initialize the ROS node
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)
        rospy.loginfo("Monitoring robot status continuously...")
        # rospy.spin()  # This keeps the node alive and processing incoming messages

    def read_pose_from_file(self, filename):
        with open(filename, 'r') as file:
            parsed_data = yaml.safe_load(file)
            # data = file.read()

        # Manually parse the file content
        # parsed_data = eval(data)
        
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



    def publish_goal(self, response):


        while True:
            # checking if:
            if self.breaking == False:
                print(response.choices[0].message.parsed.sequence)
                print(response.choices[0].message.parsed.reason)
                is_valid = (input(f"Is the sequence valid? \n"))
            else:
                is_valid = "n"

            #  if not valid, re-query llm
            if is_valid!="y":
                print(f"logs: \n {self.llm.logs}")
                response = self.llm.get_response()
                self.breaking = False

            else: 
                # is_valid is y
                self.breaking = False
                print("Executing")
                tasks = response.choices[0].message.parsed.sequence

                for task in tasks:
                    print(f"performing task {task}") # self.taskprogress_status  = running
                    print(self.tb_status)

                    # navigation
                    if task in self.loc_options:
                        print(f"Turtlebot is going to {task}")
                        selected_pose = self.pose_dict[task]
                        goal = PoseStamped()
                        goal.header = selected_pose.header
                        goal.pose = selected_pose.pose.pose
                        for i in range(2):
                            self.goal_pub.publish(goal)
                            time.sleep(1)

                    # manipulation
                    elif task in self.arm_options:
                        print(f"Arm is {task}")
                        if task == "pickup":
                            self.mygello.pickup()
                            while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                                if self.llm.get_vlm_feedback_gripper(task)==1:
                                    break
                                time.sleep(0.5)
                            self.mygello.pickup_complete()

                        elif task == "dropoff":
                            self.mygello.dropoff()
                            while self.vlm_for_gripper: # only uses vlm to complete pickup, is this param=1
                                if self.llm.get_vlm_feedback_gripper(task)==1:
                                    break
                                time.sleep(0.5)
                            self.mygello.dropoff_complete()

                    # option to wait -- only for navigation
                    elif task=="wait":
                        self.cancel_pub.publish(GoalID())
                        print("Robot waiting..")

                    while self.tb_status!=3:
                        key_= input("Press 'y':Continue 'q':Break")
                        time.sleep(1)
                        print("waiting done")
                        
                        if key_== "q":
                            print("Breaking ....")
                            self.breaking = True
                            break
                        else:
                            continue
                    
                    if self.breaking== True:
                        break
                    else:
                        self.llm.logs += f"Task progress: {task} executed successfully" # task ended
                        print(f"log -- {self.llm.logs}")

                if self.breaking==False:
                    print("Job done :) Bye") 

                            

    def check_code(self):
        while True:
            print(self.tb_status)
            # time.sleep(1)


if __name__ == "__main__":
    try:
        goal_setter = RandomGoalSetter()
        goal_setter.llm.connection_check()
        # response = goal_setter.llm.get_response()
        # goal_setter.publish_goal(response)
        for i in range(4):
        # while True:
            # print(goal_setter.llm.get_vlm_feedback_gripper("pickup"))
            time.sleep(1)
            goal_setter.subtask_name.publish("Z")


        while True:
            goal = str(input("Enter str"))
            goal_setter.subtask_name.publish(goal)


        # goal_setter.check_code()
        # rospy.sleep(3)
    except rospy.ROSInterruptException:
        print('Exception Occured')
        pass 

    
