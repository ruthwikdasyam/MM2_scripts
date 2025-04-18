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
    def __init__(self, loc_options=['ruthwik', 'zahir', 'amisha', 'kasra'], arm_options=["pickup", "dropoff"]):

        self.logs=""
        self.loc_options = loc_options
        self.arm_options = arm_options

        # dict contaning functions and things it needs to execute
        self.robots_actions = {
            "navigate_to_person":["person_name"],
            "navigate_to_point":["x","y","z","w1","w2","w3","w4"],
            "navigate_to_object":["object_name"],
            "manipulate":["pickup or place"],
            "ask_user":["question"],
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


    def get_vlm_feedback(self, task, rs_image):
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
            From the image provided, generate a detailed and structured caption describing:
            The objects present (including their types, colors, relative sizes, and locations)
            Any people or animals in the scene and their activities or interactions
            The layout or structure of the environment (e.g., indoors/outdoors, room type, background details)
            Any actions or events taking place
            Possible affordances or interactions the robot could perform with objects
            Relevant semantic context that would help the robot remember or reason about this scene later
            Be concise, yet include all important observations that would help a robot perceive, remember, and act in this environment.
            """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": f"Analyze the image and decide the correct action for the '{task}' task."},
                        {"type": "text", "text": f"Understand the image and caption it"},
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

        self.logs = self.get_text_from_jsonl()
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
        The robot has memory, vision, and manipulation abilities, and it continuously logs its experiences.
        ---
        ### Robot Memory
        You have access to the robot's memory 
        {self.logs}
        It is a list of relevant logs at a given timestamp. At each timestamp, it has:
        - Robot status - robot's base_status, arm_status, base_position and arm_position
        - llm - User queries, corresponding LLM responses and reasoning
        - camera_observation - description of what the robot had seen from that point
        - task_progress - task_name that is executed and task_status
        Use this memory to reason about the context and generate smarter plans for this robot. 
        ---
        ### Robot Capabilities
        The robot can:
        - Navigate to people or predefined places: **{self.loc_options}**
        - Move to a specific (x,y,z,w,x,y,z) base_position
        - Approach objects (identified visually)
        - Capture images and return captions of its surroundings
        - Use its manipulator to pick/place objects
        - Communicate with users and ask for clarifications
        - Wait/idle and observe without taking action
        ---
        ### Your Objective
        Given a **user query**, your job is to:
        1. Understand the user's intent.
        2. Review the robot's memory to inform your response.
        3. Generate a descriptive plan based on what user needs for the robot to execute using its abilities and past experiences.
        ---
        -Respond with a plan and a reason why the plan should work.
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
        task_sequence_schema = {
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


        # **Step 1: Understanding the user query**
        # Make sure you're using the chat.completions.create
        sequence_response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo"
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are generating a sequence of tasks for the mobile manipulator robot.
                    An assistant has generated a plan for the robot to follow. 
                    Your job is to generate the task sequence which should be in {self.robots_actions}.
                    Make sure to define the parameters required for each task.
                    """
                },
                {
                    "role": "user",
                    "content": f"Robots plan: {plan} Reason: {reason}."
                }
            ],
            # response_format="schema",  # key part!
            # schema=task_sequence_schema   # NOT `response_format_schema`
            )

        # response_2 = sequence_response.choices[0].message.content
        # respose generated by this function is a sequence of tasks that the robot should follow.
        # This should be in dict format with
        #    Keys: task_name
        #    Values: parameters
        # return response_2
        return



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
                "content":  f"Extract all relevant and related keywords from the following user query. "
                            f"These keywords will be used to search robot experiences that include image captions, tasks, task statuses, "
                            f"The goal is to retrieve all experiences that are possibly relevant "
                            f"to the user query's intent. Extract keywords that include key objects, actions, locations, task-related terms, as well as related synonyms, "
                            f"paraphrases, and contextual variations. Be exhaustive to ensure broad matching.\n\n"
                            f"User query: '{user_query}'\n\n"
                            f"Provide only a comma-separated list of keywords and phrases, without any explanations."
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
                        experience["task_progress"]["task_status"]
                    ]
                elif experience["type"]=="llm":
                    text_fields = [
                        experience["llm"]["user_query"],
                        experience["llm"]["response"],
                        experience["llm"]["reasoning"]
                    ]

                # Check if any keyword appears in the text fields
                if any(keyword.lower() in " ".join(text_fields).lower() for keyword in keywords):
                    filtered_experiences.append(experience)

        # Save the filtered experiences to a new JSONL file
        with open(output_file, "w") as outfile:
            for exp in filtered_experiences:
                json.dump(exp, outfile)
                outfile.write("\n")  # Ensure each experience is on a new line

        print(f"Filtered {len(filtered_experiences)} experiences and saved to {output_file}")


    def get_response_with_memory(self):

        query = input("Hey, how can I help you?\n")
        keywords = self.generate_keywords(query)
        print(f"\nExtracted Keywords: {keywords}")
        self.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
        response = self.get_response(user_query=query)
        print(f"\n{response.plan}")
        print(f"\n{response.reason}")
        response_2 = self.get_response_sequence(plan=response.plan, reason=response.reason)
        print("\nTask Sequence ------")
        print(f"{response_2.sequence}")
        print(f"\n{response_2.reason}")
        return response


if __name__ == "__main__":

    llm = LanguageModels()
    llm.connection_check()
    # Example usage
    # user_query = "Can you pick up the red ball from the table?"
    # user_query = input("Enter your query: ")
    # keywords = llm.generate_keywords(user_query)
    # print(f"Extracted Keywords: {keywords}")
    # llm.filter_experiences("memory_files/robot_logs.jsonl", "memory_files/filtered_experiences.jsonl", keywords.split(","))
    
    # response = llm.get_response(user_query=user_query)
    # print(response.plan)
    # print("\n")
    # print(response.reason)

    llm.get_response_with_memory()
    