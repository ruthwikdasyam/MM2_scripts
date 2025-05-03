from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List
from dataclasses import dataclass, field
import base64
import cv2
import io
from PIL import Image
import json
from typing import List, Dict, Union
from datetime import datetime, timedelta
import time
import os

client = OpenAI()

""" High level thinking - 1st step """
class ReasonResponse(BaseModel):
    plan: str
    reason: str

""" High level thinking - 2nd step """
class GripperAction(BaseModel):
    action: int = Field(..., ge=0, le=1, description="0 for open, 1 for close")
    reason: str


class LanguageModels:
    def __init__(self, loc_options=['ruthwik', 'zahir', 'amisha', 'kasra'], arm_options=["start_pickup","complete_pickup","start_dropoff","complete_dropoff"]):
        
        self.logs=""
        self.loc_options = loc_options
        self.arm_options = ["start_pickup","complete_pickup","start_dropoff","complete_dropoff"]

        # dict contaning functions and things it needs to execute
        self.robots_actions = {
                    "navigate_to_person":[f"one person_name from {self.loc_options} only", 
                                          "WHEN TO USE: When you are clear that the robot should go to this person", 
                                          "HOW IT WORKS: Robot has pre-defined locations of these persons, and the robot directly navigates there"],
                    "navigate_to_position":["x","y","z","w1","w2","w3","w4", 
                                            "WHEN TO USE: When its clear from memory that the object or whatever robot is looking for, can be seen or found here", 
                                            "HOW IT WORKS: These coordinates are direcly gicen to the move_base and robot navigates there"],
                    "navigate_to_object": ["Go to object that robot can currently see", 
                                           "WHEN TO USE: When you have arrived at a point from where object is visible, but you need to move close to it", 
                                           "HOW IT WORKS:  Robot identifies the object from it camera first, and moves to that object untill it reaches a threshold distance"],
                    "manipulate":[f"one function_name from {self.arm_options} only", 
                                  "WHEN TO USE: When you have to pickup or dropoff something, and this small arm and gripper should be able to hold it ", 
                                  "HOW IT WORKS: Gripper reaches out to the object in its workspace and grips it, then places of again when requested"],
                    "get_image_caption":["prompt", 
                                         "WHEN TO USE: When there is something that you want to know from the current camera image", 
                                         "HOW IT WORKS: The Image along with the prompt goes to VLM and returns response accordingly"],
                    "ask_user":["whatever you wanna ask or tell", 
                                "WHEN TO USE: When you need to ask/tell something or need help doing something for you. Ask only when needed", 
                                "HOW IT WORKS: This question will be displayed to user, and the user enters the response in return"],
                    "wait":["No parameter",
                            "WHEN TO USE: When robot has to stop everything and just wait",
                            "HOW IT WORKS: All current running tasks are interrupted and stopped"],
                    }
    
        self.memory_format = {
                    "base_position": ["The x,y,z,w1,w2,w3,w4 of the mobile base at that instant"],
                    "arm_position": ["Arm position at that instant"],
                    "camera_observation": ["Captions for what the robot sees through its camera"],
                    # Task Progress
                    "task_name": ["Current task that is active"],
                    "parameter": ["Required parameter for this respective task"],
                    "task_status": ["Status on the task, whether its running or completed"],
                    "task_info": ["This is a optional slot, Used for few tasks to convey related information"],
                    # LLM Logs
                    "user_input": ["Input from the user"],
                    "response": ["Language Models response on what should robot do"],
                    "reasoning": ["Reason behind the response plan it generated"],
                    "sequence": ["Sequence of tasks for the robot to execute"],
                    }

        self.save_folder = '/home/nvidia/catkin_ws/src/nav_assistant/scripts/Experiments/Ex6/Images'

    def connection_check(self):
        print("Connected")


    # Convert OpenCV image (frame) to base64
    def get_encoded_image_realsense(self, rs_image, target_size=(256, 256), jpeg_quality=70):
        if rs_image is not None:
            resized_image = cv2.resize(rs_image, target_size, interpolation=cv2.INTER_AREA)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            _, buffer = cv2.imencode(".jpg", resized_image, encode_param)
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            
            return img_base64


    def get_vlm_feedback(self, task, rs_image, question=None):
        if question is not None:
            print("writing images")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.save_folder, f"{timestamp}.png")
            cv2.imwrite(filename, rs_image)


        encoded_image = self.get_encoded_image_realsense(rs_image=rs_image)  # Convert OpenCV image to base64
        
        # Define task-specific instructions
        if task == "pickup":
            system_prompt = """
            You are an assistant for a robotic manipulator.
            ### Robot:
            - The manipulator can open and close its gripper.
            ### Task:
            Analyze an image and decide if the gripper can successfully close around an object.
            ### Rules:
            1. Respond with a reason first, then strictly output "1" (can close) or "0" (cannot).
            2. The object must be fully between the gripper fingers to count as "1".
            """
        elif task == "dropoff":
            system_prompt = """
            You are an assistant for a robotic manipulator.
            Decide if the gripper should open based on an image.
            Rules:
            - Output a reason, then "1" (if a person securely holds the object) or "0" (otherwise).
            Format:
            Reason (short). Then "1" or "0" on a new line.
            """
        elif task == "caption":
            system_prompt = """
            Assist a mobile manipulator by writing a short caption summarizing a scene.
            Include: object types/colors/sizes/locations, people/animal actions, environment structure/events, possible robot interactions, key semantic info.
            Be concise and descriptive.
            """
        elif task == "caption_2":
            system_prompt = f"""
            You assist a robotic mobile manipulator in understanding its environment.
            Task:
            Based on the provided image, generate a short caption answering this question: {question}
            Focus only on what the robot needs to know. Be clear and concise.
            """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Understand the Image and Respond Accordingly"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]}
            ],
            max_tokens=300,    # small limit for short tasks
            timeout=10         # avoid hanging too long
        )
        # Extract the response
        result = response.choices[0].message.content
        return result


    def get_response(self, user_query=None):
        
        # get query from user
        if user_query is None:
            self.user_query = input("Hey, how can I help you?\n")
        else:
            self.user_query = user_query
        self.filtered_experiences = self.get_text_from_jsonl(file_path="memory_files/filtered_experiences.jsonl")
        self.recent_experiences = self.get_text_from_jsonl(file_path="memory_files/recent_experiences.jsonl")

        # **Step 1: Understanding the user query**
        reason_response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are the brain for a **mobile manipulator robot assistant** that helps users.
                        You need to keep planning and executing tasks based on user queries, robot experiences, and progress.

                        ---
                        ### Robot Memory
                        This should be used only to take help to generate parameters for the tasks, and get help contextually on the environment.
                        You have access to the robot's memory.
                        It is a list of relevant logs at a given timestamp. At each timestamp, it has: {self.memory_format}
                        Use this memory {self.filtered_experiences} to reason about the context and generate smarter plans for this robot. 
                        ---
                        ### Robot Logs
                        This should be used for designing the task sequence, which means generating further plan.
                        You also have access to its live current logs, which is recent robots memory on the task its running.
                        Use this to check task related information to understand the progress and plan further, (if progressed).
                        So, from the logs if it completes a task, it should be moved to next task, **only use it to check task_status**. 
                        Here are current logs {self.recent_experiences} in format {self.memory_format}
                        ---
                        ### Robot Capabilities
                        The robot can:
                        - Navigate to only these people: **{self.loc_options}**
                        - Move to a specific position (x,y,z,w,x,y,z) base_position, use this if you know the location.
                            or if you cant see that object clearly, navigating there should help.
                        - Use its manipulator to pick/place small objects and has only options to **{self.arm_options}**. 
                          Its gripper is small and only can carry objects like pens, ball, cup and similar objects of sizes and weight.
                            - to pick it can start_pickup, then, it performs complete_pickup
                            - to pick it can start_dropoff, it performs complete_dropoff
                        - Robot sees through the camera on it. Capture images and return info
                        - Communicate with users. You can ask questions, ask for help, tell anything and more, ONLY IF NECESSARY
                        - Wait/idle stopping everything that you are doing
                        ---
                        ### Your Objective
                        Given a **user query**, **current progress**, and **Memory** your job is to:
                        1. Understand the user's intent and task progress.
                        2. Review the robot's memory to inform your response.
                        3. Goal is to understand user query and plan further for next steps based on its progress.
                            if robot needs to move, Generate a descriptive plan based on what user needs robot to do, and what progress robot has completed using its progress.
                        4. Remember that if you can answer user input just from the the info you have, you can just answer it without any robot action. 
                            You should only move and utilize functions when its needed.
                        ---
                        Respond with a plan that robot should further implement, and a reason why the plan should work. 
                        ---
                        """
                },
                {
                    "role": "user",
                    "content": f"{self.user_query}."
                }
            ],
            response_format=ReasonResponse,
        )
        reason_text = reason_response.choices[0].message.parsed
        sequence_response = reason_text
        return sequence_response


    def get_response_sequence(self, plan, reason):
        # **Step 1: Understanding the user query**
        sequence_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are generating a sequence of tasks for the mobile manipulator robot should perform.
                    An assistant has generated a plan that robot should further perform for the robot to follow based on how much it progressed, which is shown here: {self.recent_experiences}
                    Your job is to generate the task sequence with appropriate parameters, that robot should futher perform which should be in {self.robots_actions}.
                    If its running, means same task should run untill it completes. If a task is shown completed or succeeded, it means the plan should be generated to perform from next task.
                    Once the task is completed, it means it shouldnt be shown in the plan.
                    Make sure to define the parameters required for each task. You are allowed to directly use the base_position and arm_position from the robot's status in the previous experience if you plan to navigate to that specific location.
                    Respond with a JSON format only.
                    """
                },
                {
                    "role": "user",
                    "content": f"Robots plan: {plan} Reason: {reason}."
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_task_sequence",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task": {"type": "string"},
                                        "parameter": {"type": "string"}
                                    },
                                    "required": ["task", "parameter"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["steps"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
            )
        
        response_2 = sequence_response.choices[0].message.content
        return  response_2


    # PRIVATE METHODS
    def get_text_from_jsonl(self, file_path="memory_files/filtered_experiences.jsonl"):
        texts = []
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line:  # skip empty lines
                    texts.append(line)
        return texts


    def generate_keywords(self, user_query):

        # Using the new API method for text generation
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content":  f"""
                            Extract all relevant and related **one word** keywords from the following user query.
                            These keywords will be used to search robot experiences that include image captions, tasks, task statuses,
                            The goal is to retrieve all experiences that are relevant to the user query. Extract keywords that include context, object rekated terms
                            Be exhaustive to ensure broad matching.\n
                            User query: '{user_query}'\n\n
                            Provide only a comma-separated list of keywords, without any explanation.
                            Need to keep number of keywords to a minimum in range of 10 - 20.
                            Return them in the order of priority, important keywords first.
                            """
            }],
            temperature=0.5
        )
        # Extract the keywords from the response
        keywords = response.choices[0].message.content
        return keywords


    def filter_experiences(self, input_file, output_file, keywords):
        """
        Filters robot experiences based on relevant keywords and saves them to a new JSONL file.
        """
        filtered_experiences = []
        limit = 100

        with open(input_file, "r") as infile:
            for line in infile: 
                experience = json.loads(line)
                
                if experience["type"]=="status":
                    # Extract relevant fields for keyword matching
                    text_fields = [
                        experience["camera_observation"],
                        # experience["task_progress"]["task_name"],
                        # experience["task_progress"]["parameter"],
                        # experience["task_progress"]["task_status"],
                        experience["task_progress"]["task_info"]
                    ]
                elif experience["type"]=="llm":
                    text_fields = [
                        experience["llm"]["user_input"],
                        experience["llm"]["response"],
                        experience["llm"]["reasoning"]
                        # experience["llm"]["sequence"]
                    ]

                # Check if any keyword appears in the text fields
                if any(keyword.lower() in " ".join(text_fields).lower() for keyword in keywords):
                    # Limit the number of experiences to 100
                    if len(filtered_experiences) < limit:
                        filtered_experiences.append(experience)
                    else:
                        break
        # Save the filtered experiences to a new JSONL file
        with open(output_file, "w") as outfile:
            for exp in filtered_experiences:
                json.dump(exp, outfile)
                outfile.write("\n")  # Ensure each experience is on a new line

        print(f"Filtered {len(filtered_experiences)} experiences")



    def get_recent_20_experiences(self, input_file, output_file, time_window_minutes=5, max_experiences=30, newtask_time = None):
        """
        Collects the last up to `max_experiences` experiences from the log file,
        only if they are within `time_window_minutes` of the latest timestamp.
        """
        if newtask_time:
            time_window_minutes = min((time.time() - newtask_time)/60, time_window_minutes)
            # print(time_window_minutes)
            # print((time.time() - newtask_time)/60)
        all_experiences = []
        # First pass: load all valid timestamped experiences
        with open(input_file, "r") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                exp = json.loads(line)
                timestamp_str = exp.get("timestamp") or exp.get("tamp")  # in case of typo
                if timestamp_str:
                    try:
                        exp_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        exp["__parsed_timestamp"] = exp_time
                        all_experiences.append(exp)
                    except ValueError:
                        continue  # skip malformed timestamp entries

        if not all_experiences:
            print("No timestamped experiences found.")
            return
        # Find the latest timestamp
        # latest_time = max(exp["__parsed_timestamp"] for exp in all_experiences)
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=time_window_minutes)
        # Go from end, collecting only those within the cutoff time
        recent_experiences = []
        for exp in reversed(all_experiences):
            if exp["__parsed_timestamp"] >= cutoff_time:
                recent_experiences.append(exp)
                if len(recent_experiences) == max_experiences:
                    break
        # Write results in chronological order
        with open(output_file, "w") as outfile:
            for exp in reversed(recent_experiences):  # reverse to restore original order
                exp.pop("__parsed_timestamp", None)
                json.dump(exp, outfile)
                outfile.write("\n")

        print(f"Collected {len(recent_experiences)} recent experiences")


    def get_response_with_memory(self):
        query = input("Hey, how can I help you?\n")
        keywords = self.generate_keywords(query)
        print(f"\nExtracted Keywords: {keywords}")
        self.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
        self.get_recent_20_experiences("memory_files/robot_logs.jsonl", "memory_files/recent_experiences.jsonl")
        # response = self.get_response(user_query=query)
        # print(f"\n{response.plan}")
        # print(f"\n{response.reason}")
        # response_2 = self.get_response_sequence(plan=response.plan, reason=response.reason)
        # print("\nTask Sequence ------")
        # # print(response_2)
        # data = json.loads(response_2)
        # for i, step in enumerate(data["steps"], start=1):
        #     print(f"Step {i}: {step['action']}, = {step['parameter']}")
        # print(f"{response_2.sequence}")
        # print(f"\n{response_2.reason}")
        # return response


if __name__ == "__main__":

    llm = LanguageModels()
    llm.connection_check()

    # query = input("Hey, how can I help you?\n")
    # keywords = llm.generate_keywords(query)
    # print(f"\nExtracted Keywords: {keywords}")
    # llm.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
    # response = llm.get_response(user_query=query)
    # # print(f"\n{response.plan}")
    # # print(f"\n{response.reason}")
    # response_2 = llm.get_response_sequence(plan=response.plan, reason=response.reason)
    # print("\nTask Sequence ------")
    # # print(response_2)
    # data = json.loads(response_2)
    # for i, step in enumerate(data["steps"], start=1):
    #     print(f"Step {i}: {step['action']}, = {step['parameter']}")
    
    llm.get_response_with_memory()
