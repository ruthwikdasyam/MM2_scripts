#!/usr/bin/env python


import rospy
from std_msgs.msg import String
import time

class MemoryNode:

    def __init__(self):

        # Initialize the ROS node
        rospy.init_node('memory_node', anonymous=True)    
        rospy.Subscriber('subtask', String, self.callback)
        # rospy.spin()

        self.tname = " "

    # Callback function for the subscriber
    def callback(self, data):
        rospy.loginfo("Received message: %s", data.data)
        self.tname = data.data
    



if __name__ == '__main__':
    try:
        mem_node = MemoryNode()

        while True:
            print(f"-- {mem_node.tname}")
            time.sleep(1)
    except rospy.ROSInterruptException:
        pass