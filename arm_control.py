from mobilegello.gello_controller import GELLOcontroller
import time
import numpy as np

# make_hdf5("lift_data")


mygello = GELLOcontroller("doodle", torque_start=True)


mygello.rest()
# mygello.pickup()
# mygello.dropoff()
print(mygello.read_encoder_values())