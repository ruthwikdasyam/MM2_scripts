import json 
import os
# import openai
from openai import OpenAI

location_map = json.load(open("/home/raas/nav_assistant/location_pose_map.json"))
loc_options = ', '.join(list(location_map.keys()))

print(loc_options)

user_inp = input("What do you want to do?\n")
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": f"You are a helpful agent to a robot. The robot has a map with following predefined people: {loc_options}. A user will ask for some help from the robot. Your task is to uderstand the query and tell the robot which of the persons/place the robot should go to help with the user's query. The PERSON/PLACE NAME YOU ANSWER WITH MUST BE IN THE LIST. THE NAME MUST MATCH THE NAMES YOU HAVE BEEN GIVEN. The answer should be in this format: <Person/place Name>: <Reason for choosig this>"
    },
    {
      "role": "user",
      "content": f"Here is the user's query: {user_inp}."
    }
  ],
  temperature=0.5,
  max_tokens=64,
  top_p=1
)

assistant_reponse = response.choices[0].message.content
target_room, reason = assistant_reponse.split(':')
print(f'Target room is {target_room}')
print(f'The reason is {reason}')