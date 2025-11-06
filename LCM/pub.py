import lcm
from exlcm import pose_t

msg = pose_t() # create a pose_t object
msg.x = 1.0 # set the x value
msg.y = 2.0 # set the y value
msg.theta = 0.5 # set the theta value

lc = lcm.LCM() # create a LCM object
lc.publish("POSE_CHANNEL", msg.encode()) # publish the message to the channel