from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List
from dataclasses import dataclass, field
import cv2
import base64
import io
from PIL import Image
import json
from typing import List, Dict, Union
from datetime import datetime, timedelta

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
        self.arm_options = ["start_pickup","open_gripper", "complete_pickup","start_dropoff","close_gripper","complete_dropoff"]

        # dict contaning functions and things it needs to execute
        self.robots_actions = {
            "navigate_to_person":[f"one person_name from {self.loc_options} only"],
            "navigate_to_position":["x","y","z","w1","w2","w3","w4"],
            "navigate_to_object":["object_name"],
            "manipulate":[f"one function_name from {self.arm_options} only"],
            "get_image_caption":["prompt on what you want to know"],
            "ask_user":["question"],
            "wait":[],
        }
    

    def connection_check(self):
        print("Connected")


    # Convert OpenCV image (frame) to base64
    def get_encoded_image_logitech(self, rs_image=None):
        if rs_image is not None:
            cv2.imwrite("image_rs.jpg", rs_image)
            _, buffer = cv2.imencode(".jpg", rs_image)  # Encode as JPEG
            img_base64 = base64.b64encode(buffer).decode("utf-8")  # Convert to base64 string
            return img_base64
        else:
            self.cap = cv2.VideoCapture(0)  # Open camera at index 1
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                return None
            try:
                while True:
                    ret, frame = self.cap.read()
                    if ret:
                        cv2.imwrite("image_gripper.jpg", frame)
                        break  # Exit loop once a valid frame is captured
                _, buffer = cv2.imencode(".jpg", frame)  # Encode as JPEG
                img_base64 = base64.b64encode(buffer).decode("utf-8")  # Convert to base64 string
                return img_base64
            except KeyboardInterrupt:
                print("\n[INFO] Ctrl+C detected. Exiting gracefully...")
            finally:
                self.cap.release()


    def get_encoded_image_realsense(self):
        return 


    def get_vlm_feedback(self, task, rs_image, question=None):
        # if not task=="pickup" or not task=="dropoff" or not task=="caption":
            # print("exiting")
            # print(f"Task {task} not in pickup/dropoff for Gripper")
            # return
        # get vlm response
        encoded_image = self.get_encoded_image_logitech(rs_image=rs_image)  # Convert OpenCV image to base64
        
        # Define task-specific instructions
        if task == "pickup":
            system_prompt = """
            You are an intelligent assistant for a robotic manipulator.

            ### Robot Capabilities:
            - The **manipulator** can open or close its gripper.

            ### Task:
            Determine whether the gripper can successfully close around an object by analyzing the given image.

            ### Constraints:
            1. **Respond strictly with either "1" (if it can close) or "0" (if it cannot).** and give reason
            2. The object must be positioned between the fingers of the gripper, ensuring it will be grasped when closed.
            3. the response should be reason, and the last string should be 0 or 1
            """
        elif task == "dropoff":
            system_prompt = """
            You are an intelligent assistant for a robotic manipulator.

            ### Robot Capabilities:
            - The **manipulator** can open or close its gripper.

            ### Task:
            Determine whether the gripper should open and release the object by analyzing the given image.

            ### Constraints:
            1. **Respond strictly with either "1" (if it should open) or "0" (if it should not open).** and give reason
            2. The gripper should **only open** if a person is securely holding the object.
            3. If the object is not being held by a person, the gripper **should not open** to prevent it from falling.
            """
        elif task == "caption":
            system_prompt = """
            You are assisting a robotic mobile manipulator in understanding its environment.
            From the image provided, generate a brief caption describing:
            The objects present (including their types, colors, relative sizes, and locations)
            Any people or animals in the scene and their activities or interactions
            The layout or structure of the environment.
            Any actions or events taking place
            Possible affordances or interactions the robot could perform with objects
            Relevant semantic context that would help the robot remember or reason about this scene later
            Be concise, yet include all important observations that would help a robot perceive, remember, and act in this environment.
            """

        elif task == "caption_2":
            system_prompt = f"""
            You are assisting a robotic mobile manipulator in understanding its environment.
            From the image provided, generate a brief caption describing what robot needs to know.
            Heres the question its asking: {question}
            """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": f"Analyze the image and decide the correct action for the '{task}' task."},
                        {"type": "text", "text": f"Understand the image and respond"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
        )
        # Extract the response
        result = response.choices[0].message.content
        # print(f"Model Response for {task}: {result}")

        # Convert response to integer for further processing
        # result = int(result)
        return result


    def get_response(self, user_query=None):
        
        # get query from user
        if user_query is None:
            self.user_query = input("Hey, how can I help you?\n")
        else:
            self.user_query = user_query

        self.filtered_experiences = self.get_text_from_jsonl(file_path="memory_files/filtered_experiences.jsonl")
        self.recent_experiences = self.get_text_from_jsonl(file_path="memory_files/recent_experiences.jsonl")
        # data from previous attempts
        # if self.logs == "":
        #     add_on = ""
        # else:
        # add_on = f"""
        #         """

        # print(add_on)

        # **Step 1: Understanding the user query**
        reason_response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
        You are a task planner for a **mobile manipulator robot assistant** that helps users in a shared workspace.
        You need to keep planning and executing tasks based on user queries, robot experiences, and progress.
        The robot has memory, vision, and manipulation abilities, and it continuously logs its experiences and you should plan further.
        ---
        ### Robot Memory
        You have access to the robot's memory 
        {self.filtered_experiences}
        It is a list of relevant logs at a given timestamp. At each timestamp, it has:
        - Robot status - robot's base_status, arm_status, base_position and arm_position
        - llm - User queries, corresponding LLM responses and reasoning
        - camera_observation - description of what the robot had seen from that point
        - task_progress - task_name that is executed and task_status
        Use this memory to reason about the context and generate smarter plans for this robot. 
        ---
        ### Robot Logs
        You also have access to its current logs, which is recent robot logs on the task its running.
        Use this to check "task_status" and understand the progress and plan further, if progressed. Here are current logs {self.recent_experiences}
        ---
        ### Robot Capabilities
        The robot can:
        - Navigate to only these people: **{self.loc_options}**
        - Move to a specific position (x,y,z,w,x,y,z) base_position
        - Navigate to specific object (identified visually)
        - Use its manipulator to pick/place objects and has only options to **{self.arm_options}**
            - to pick it can start_pickup, then, close_gripper, and it performs complete_pickup
            - to pick it can start_dropoff, then, open_gripper, and, it performs complete_dropoff
        - Robot sees through the camera on it. Capture images and return what you want to know about it
        - Communicate with users and ask for clarifications
        - Wait/idle stopping everything that you are doing
        ---
        ### Your Objective
        Given a **user query**, **current task logs**, and **Memory** your job is to:
        1. Understand the user's intent and task progress, **if any**.
        2. Review the robot's memory to inform your response.
        3. Generate a descriptive plan based on what user needs, and what progress robot has completed, for the robot to execute, using its abilities and progress.
        ---
        -Respond with a plan that robot should further implement, and a reason why the plan should work.
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
        
        # - Navigate to people or predefined places: **{self.loc_options}**
        # - Move to a specific (x,y,z,w,x,y,z) base_position
        # - Approach objects (identified visually)
        # - Capture images and return captions of its surroundings
        # - Use its manipulator to pick/place objects             
        # - Communicate with users and ask for clarifications
        # - Wait/idle and observe without taking action
        
        # print(f"Step 1 - Reason: {reason_text}")  # Debugging print, you can remove this

        # **Step 2: Generating the Task Sequence**
        # sequence_response = client.beta.chat.completions.parse(
        #     model="gpt-4o",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": f"""
        #             You are a mobile manipulator robot.

        #             ### Task:
        #             Your goal is to generate a **valid sequence of tasks** based on the given reason.

        #             ### Input:
        #             - User Query: {self.user_query}
        #             - Reason: {reason_text}

        #             ### Constraints:
        #             1. Use only the predefined locations/people: **{self.loc_options}**.
        #             2. Use only the predefined manipulator tasks: **{self.arm_options}**.
        #             3. Ensure a logical sequence, The generated sequence should alternate as follows:
        #                 - **[Person Name] → [Manipulator Task] → [Person Name] → [Manipulator Task] → ...**
        #             4. If the reason states that the task **cannot** be executed, return:
        #                 - `sequence: a sequence as in Constraint 3`
        #             5. Otherwise, generate a **clear and structured sequence**.

        #             {add_on}
        #             """
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Reason: {reason_text}"
        #         }
        #     ],
        #     response_format=TaskSequence,  # A structured response class for task sequences
        # )
        sequence_response = reason_text

        # self.logs = f"User input: {self.user_query}. Reason: {reason_text}. Response: {sequence_response.choices[0].message.parsed.sequence}" + self.logs

        return sequence_response


    def get_response_sequence(self, plan, reason):
        task_sequence_tool = {
            "type": "function",
            "function": {
                "name": "generate_task_sequence",
                "description": "Generate a task sequence for a mobile manipulator robot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task_name": { "type": "string" },
                                    "parameters": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "oneOf": [
                                                { "type": "string" },
                                                { "type": "number" },
                                                { "type": "array", "items": { "type": ["string", "number"] } }
                                            ]
                                        }
                                    }
                                },
                                "required": ["task_name", "parameters"]
                            }
                        }
                    },
                    "required": ["tasks"]
                }
            }
        }

        # self.filtered_experiences = self.get_text_from_jsonl(file_path="memory_files/filtered_experiences.jsonl")
        # self.recent_experiences = self.get_text_from_jsonl(file_path="memory_files/recent_experiences.jsonl")

        # **Step 1: Understanding the user query**
        # Make sure you're using the chat.completions.create
        sequence_response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo"
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are generating a sequence of tasks for the mobile manipulator robot should perform.
                    An assistant has generated a plan that robot should further perform for the robot to follow based on these robots relavant previous experiences {self.filtered_experiences} and current task logs {self.recent_experiences}.
                    Your job is to generate the task sequence with appropriate parameters, that robot should futher perform which should be in {self.robots_actions}.
                    Make sure to define the parameters required for each task. You are allowed to directly use the base_position and arm_position from the robot's status in the previous experience if you plan to navigate to that specific location.
                    Respond with a JSON format only.
                    """
                },
                {
                    "role": "user",
                    "content": f"Robots plan: {plan} Reason: {reason}."
                }
            ],
            # response_format=task_sequence_schema,  # key part!
            # schema=task_sequence_schema   # NOT `response_format_schema`
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
        # "final_answer": {"type": "string"}
        response_2 = sequence_response.choices[0].message.content
        # respose generated by this function is a sequence of tasks that the robot should follow.
        # This should be in dict format with
        #    Keys: task_name
        #    Values: parameters
        # return response_2
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
                            Extract all relevant and related keywords from the following user query.
                            These keywords will be used to search robot experiences that include image captions, tasks, task statuses,
                            The goal is to retrieve all experiences that are possibly relevant to the user query's intent. Extract keywords that include task-related terms, as well as related synonyms,
                            Be exhaustive to ensure broad matching.\n
                            User query: '{user_query}'\n\n
                            Provide only a comma-separated list of keywords and phrases, without any explanation
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

        :param input_file: Path to the input .jsonl file containing all experiences.
        :param output_file: Path to the output .jsonl file to save filtered experiences.
        :param keywords: A list of keywords to filter experiences.
        """
        filtered_experiences = []

        with open(input_file, "r") as infile:
            for line in infile: 
                experience = json.loads(line)
                
                if experience["type"]=="status":
                    # Extract relevant fields for keyword matching
                    text_fields = [
                        experience["camera_observation"],
                        experience["task_progress"]["task_name"],
                        experience["task_progress"]["parameter"],
                        experience["task_progress"]["task_status"]
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
                    filtered_experiences.append(experience)

        # Save the filtered experiences to a new JSONL file
        with open(output_file, "w") as outfile:
            for exp in filtered_experiences:
                json.dump(exp, outfile)
                outfile.write("\n")  # Ensure each experience is on a new line

        print(f"Filtered {len(filtered_experiences)} experiences")



    def get_recent_20_experiences(self, input_file, output_file, time_window_minutes=5, max_experiences=20):
        """
        Collects the last up to `max_experiences` experiences from the log file,
        only if they are within `time_window_minutes` of the latest timestamp.

        :param input_file: Path to the input .jsonl file.
        :param output_file: Path to write the filtered recent experiences.
        :param time_window_minutes: Only include experiences from the last N minutes.
        :param max_experiences: Max number of recent experiences to include.
        """
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
        latest_time = max(exp["__parsed_timestamp"] for exp in all_experiences)
        cutoff_time = latest_time - timedelta(minutes=time_window_minutes)

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


    # def get_response_with_memory(self):
        # query = input("Hey, how can I help you?\n")
        # keywords = self.generate_keywords(query)
        # print(f"\nExtracted Keywords: {keywords}")
        # self.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
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
    
    
    # publish info

    llm.get_recent_20_experiences("memory_files/robot_logs.jsonl", "memory_files/recent_experiences.jsonl")
