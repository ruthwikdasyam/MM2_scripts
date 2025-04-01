import rospy
import random
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, PoseStamped
import glob
import yaml 
import time 
import json
from mobilegello.gello_controller import GELLOcontroller
import numpy as np


class RandomGoalSetter:
    def __init__(self):
        rospy.init_node("random_goal_setter", anonymous=True)

        self.location_map = json.load(open("/home/nvidia/catkin_ws/src/nav_assistant/jsons/location_pose_map.json"))


        # print(self.pose_list)

        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=3)

    def read_pose_from_file(self, filename):   # reading this file----
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

    def publish_random_goal(self):
        loc_options = '\n'.join(list(self.location_map.keys()))
        selected_location = input(f"Please select a goal from the following locations:\n{loc_options} \n\n")
        rospy.sleep(3)
        
        selected_pose = self.read_pose_from_file( "/home/nvidia/catkin_ws/src/nav_assistant/poses/" + self.location_map[selected_location] + "_pose.txt")

        goal = PoseStamped()
        goal.header = selected_pose.header
        goal.pose = selected_pose.pose.pose

        for i in range(2):
            self.goal_pub.publish(goal)
            time.sleep(1)
        
        # rospy.loginfo("The selected goal id is ", self.location_map[selected_location] )
        rospy.loginfo("Published this goal: \n{}".format(goal))

    def pick_up(self):
        mygello = GELLOcontroller("doodle", torque_start=True)
        new_home = np.array([2068, 2010, 2094, 1885, 2124, 3150]) #rest position
        config_1 = np.array([2338, 2447, 3393, 1740, 1994, 3710]) #pick up position - gripper open
        config_2 = np.array([2338, 2447, 3393, 1740, 1994, 3100]) #pick up position - gripper closed
        mygello.goto_controlled_home(new_home)
        mygello.goto_controlled_home(config_1)
        mygello.goto_controlled_home(config_2)
        mygello.goto_controlled_home(new_home)

if __name__ == "__main__":
    try:
        goal_setter = RandomGoalSetter()
        goal_setter.publish_random_goal()
        rospy.sleep(3)
        goal_setter.pick_up()
        rospy.sleep(3)
    except rospy.ROSInterruptException:
        print('Exception Occured')
        pass 

    
