import lcm
from exlcm import pose_t

def my_handler(channel, data):
    msg = pose_t.decode(data) # decode the message
    print(f"Received on {channel}: x={msg.x}, y={msg.y}")

lc = lcm.LCM() # create a LCM object
lc.subscribe("POSE_CHANNEL", my_handler) # subscribe to the channel

while True:
    lc.handle() # handle the message
