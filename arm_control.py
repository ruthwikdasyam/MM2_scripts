from mobilegello.gello_controller import GELLOcontroller
import time
import numpy as np

# make_hdf5("lift_data")


mygello = GELLOcontroller("doodle", torque_start=True)


# mygello.rest()
# mygello.pickup()
# mygello.dropoff()
print(mygello.read_encoder_values())


# print(mygello.read_camera_encoder_values())

# mygello.camera_home()
# mygello.camera_turn_right()
# mygello.camera_turn_up()
# mygello.camera_turn_upright()

# for i in range(100):
#     mygello.camera_home()
#     time.sleep(3)
#     mygello.camera_turn_upright()
#     time.sleep(3)
#     mygello.camera_home()
#     time.sleep(3)
#     mygello.camera_turn_upleft()
#     time.sleep(3)
    # mygello.camera_turn_up()
    # time.sleep(1)
    # mygello.camera_home()
    # time.sleep(1)
    # mygello.camera_turn_down()
    # time.sleep(1)

mygello.rest()
print(mygello.is_near_target("home"))

mygello.pickup()
time.sleep(1)
print(mygello.is_near_target("pickup"))
mygello.pickup_complete()
time.sleep(1.5)
print(mygello.is_near_target("pickup_complete"))
mygello.dropoff()
time.sleep(1)
print(mygello.is_near_target("dropoff"))
mygello.dropoff_complete()
time.sleep(1.5)
print(mygello.is_near_target("dropoff_complete"))
