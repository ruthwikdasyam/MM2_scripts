from mobilegello.gello_controller import GELLOcontroller
import time
import numpy as np

# make_hdf5("lift_data")


mygello = GELLOcontroller("doodle", torque_start=True)


mygello.rest()
# mygello.pickup()
# mygello.dropoff()
print(mygello.read_encoder_values())


print(mygello.read_camera_encoder_values())

# mygello.camera_home()
# mygello.camera_turn_right()
# mygello.camera_turn_left()

for i in range(10):
    mygello.camera_home()
    time.sleep(1)
    mygello.camera_turn_left()
    time.sleep(1)
    mygello.camera_home()
    time.sleep(1)
    mygello.camera_turn_right()
    time.sleep(1)
