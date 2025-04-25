#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import time
from actionlib_msgs.msg import GoalStatusArray
from mobilegello.gello_controller import GELLOcontroller 


class SimpleMover:
    def __init__(self):
        rospy.init_node('simple_mover_node', anonymous=True)

        # Use the teleop input of cmd_vel_mux
        self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
        self.objectnav_sub = rospy.Subscriber('/objectnav/status', GoalStatusArray, self.objectnav_callback )
        self.mygello = GELLOcontroller("doodle", torque_start=True)

        self.cmd = Twist()

        rospy.sleep(1)

    def objectnav_callback(self, msg):
        try:
            self.status = msg.status_list[-1].status
            print(self.status)
        except:
            pass

    def move_forward(self, lin_speed=0.2, ang_speed=0.0, duration=4.0, stop = True):
        self.cmd.linear.x = lin_speed
        self.cmd.angular.z = ang_speed
        start_time=time.time()
        rospy.loginfo(f"Moving forward: lin_speed={lin_speed}, ang_speed={ang_speed}, duration={duration}s")
        while time.time() - start_time <= duration:
            self.vel_pub.publish(self.cmd)
        # rospy.sleep(duration)
        if stop == True:
            self.stop()

    def stop(self):
        rospy.loginfo("Stopping robot")
        self.cmd = Twist()
        self.vel_pub.publish(self.cmd)


if __name__ == '__main__':
    try:
        mover = SimpleMover()

        # mover.mygello.camera_home()
        # mover.mygello.camera_turn_up()
        # time.sleep(2)
        # mover.mygello.camera_home()
        # mover.move_forward(0.1, 0.0, 2, False)
        # mover.move_forward(0.2, 0.0, 3, False)
        # mover.mygello.camera_turn_up()
        # mover.move_forward(0.2, -0.3, 3, False)
        # mover.mygello.camera_home()
        # mover.move_forward(0.2, 0.0, 3, False)
        # # manip start
        # mover.move_forward(0.2, -0.3, 2)
        # # manip functions
        # mover.mygello.camera_turn_up()
        # time.sleep(1)
        # mover.mygello.camera_turn_upleft()
        # time.sleep(1)
        # mover.mygello.dropoff()
        # time.sleep(1)
        # mover.mygello.open_gripper()
        # mover.mygello.dropoff_complete()
        # mover.mygello.camera_home()
        # time.sleep(1)
        # mover.mygello.camera_turn_down()
        # time.sleep(4)
        # mover.mygello.camera_home()

        # print(mover.status)

        while rospy.is_shutdown():
            try:
                if mover.status ==3:
                    mover.mygello.dropoff()
                    mover.mygello.open_gripper()
                    mover.mygello.dropoff_complete()
                    break
                else:
                    time.sleep(1)

            except rospy.ROSInterruptException:
                pass

        # rospy.spin()

    except rospy.ROSInterruptException:
        pass
