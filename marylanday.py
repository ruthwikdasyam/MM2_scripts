#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

class SimpleMover:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('simple_mover_node', anonymous=True)

        # Publisher to /cmd_vel
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Twist message
        self.cmd = Twist()

        # Wait for publisher to be ready
        rospy.sleep(1)

    def move_forward(self, speed=0.2, duration=2.0):
        self.cmd.linear.x = speed
        self.cmd.angular.z = 0.0
        rospy.loginfo(f"Moving forward: speed={speed}, duration={duration}s")
        self.vel_pub.publish(self.cmd)
        rospy.sleep(duration)
        self.stop()

    def stop(self):
        rospy.loginfo("Stopping robot")
        self.cmd = Twist()  # Zero everything
        self.vel_pub.publish(self.cmd)

if __name__ == '__main__':
    try:
        mover = SimpleMover()
        mover.move_forward()
    except rospy.ROSInterruptException:
        pass
