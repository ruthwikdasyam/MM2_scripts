from mobilegello.gello_controller import GELLOcontroller
import time
import numpy as np
import rospy
from std_msgs.msg import String, Float64MultiArray



# mygello = GELLOcontroller("doodle", torque_start=True)

class GelloCameraController:
    def __init__(self):
        self.gello = GELLOcontroller("doodle", torque_start=True)
        self.gello.camera_home()  # Initialize camera position
        rospy.loginfo("GELLO camera controller node initialized.")

        # Create subscriber
        rospy.Subscriber("/camera_command", Float64MultiArray, self.camera_command_callback)
    
    def camera_command_callback(self, msg):
        if len(msg.data) != 2:
            rospy.logwarn("Expected 2 numbers but got a different length, Returning to home position.")
            self.gello.camera_home()
            return
        

        x, y = msg.data
        print(msg.data)
        camera_values = self.gello.read_camera_encoder_values()

        goal_values = np.array([x, y]) + camera_values

        self.gello.camera_set_position(int(goal_values[0]), int(goal_values[1]))
        rospy.loginfo(f"Camera position set to: {goal_values}")


if __name__ == "__main__":
    rospy.init_node("gello_camera_controller", anonymous=True)
    controller = GelloCameraController()
    
    # Keep the node running
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down camera controller.")
