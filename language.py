from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List
from dataclasses import dataclass, field
import cv2
import base64
import io
from PIL import Image

client = OpenAI()

class ReasonResponse(BaseModel):
    reason: str

class TaskSequence(BaseModel):
    sequence: List[str]
    reason: str

class GripperAction(BaseModel):
    action: int = Field(..., ge=0, le=1, description="0 for open, 1 for close")
    reason: str

class LanguageModels:
    def __init__(self, loc_options=None, arm_options=None):

        self.logs=""
        self.loc_options = loc_options
        self.arm_options = arm_options


    
    def connection_check(self):
        print("Connected")


    # Convert OpenCV image (frame) to base64
    def get_encoded_image(self):
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


    def get_vlm_feedback(self, task):
        # if not task=="pickup" or not task=="dropoff" or not task=="caption":
            # print("exiting")
            # print(f"Task {task} not in pickup/dropoff for Gripper")
            # return
        # get vlm response
        encoded_image = self.get_encoded_image()  # Convert OpenCV image to base64
        
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
            You are assisting a robotic Mobile Manipulator.
            From the image provided to you, describe what do you find.
            This description will be used in learning about the environment and thus responding better to user queries.
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


    def get_response(self):
        
        # get query from user
        self.user_query = input("Hey, how can I help you?\n")

        # data from previous attempts
        if self.logs == "":
            add_on = ""
        else:
            add_on = f"This is the log for the previous attempt, and the robot was interrupted. {self.logs} So, consider this and plan again."

        # **Step 1: Understanding the user query**
        reason_response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a mobile manipulator robot.

                    ### Robot Capabilities:
                    - The **mobile robot** can navigate only to these predefined locations/people: **{self.loc_options}**. It **cannot** go anywhere else.
                    - The **manipulator** can perform these predefined tasks: **{self.arm_options}**.
                    - It also has a 'wait' option, stopping everything and waiting for further instructions.
                    - The robot can also go 'home' to sleep.

                    ### Task:
                    Your goal is to **understand what the user wants the robot to do** and return only the reason. You do not need to generate a task sequence yet.

                    ### Constraints:
                    1. Extract the **main task** from the user's input.
                    2. If the task requires movement, identify **where** it should go.
                    3. If it requires manipulation, identify **what** the manipulator needs to do.
                    4. If the user requests an **invalid action** (unavailable location, person, or task), return:
                        - `reason: Reason for the failure`
                    5. **DO NOT** return the task sequence at this stage.

                    {add_on}
                    """
                },
                {
                    "role": "user",
                    "content": f"{self.user_query}."
                }
            ],
            response_format=ReasonResponse,  # A structured class that stores only 'reason'
        )

        reason_text = reason_response.choices[0].message.parsed.reason
        
        print(f"Step 1 - Reason: {reason_text}")  # Debugging print, you can remove this

        # **Step 2: Generating the Task Sequence**
        sequence_response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a mobile manipulator robot.

                    ### Task:
                    Your goal is to generate a **valid sequence of tasks** based on the given reason.

                    ### Input:
                    - User Query: {self.user_query}
                    - Reason: {reason_text}

                    ### Constraints:
                    1. Use only the predefined locations/people: **{self.loc_options}**.
                    2. Use only the predefined manipulator tasks: **{self.arm_options}**.
                    3. Ensure a logical sequence, The generated sequence should alternate as follows:
                        - **[Person Name] → [Manipulator Task] → [Person Name] → [Manipulator Task] → ...**
                    4. If the reason states that the task **cannot** be executed, return:
                        - `sequence: a sequence as in Constraint 3`
                    5. Otherwise, generate a **clear and structured sequence**.

                    {add_on}
                    """
                },
                {
                    "role": "user",
                    "content": f"Reason: {reason_text}"
                }
            ],
            response_format=TaskSequence,  # A structured response class for task sequences
        )

        self.logs = f"User input: {self.user_query}. Reason: {reason_text}. Response: {sequence_response.choices[0].message.parsed.sequence}" + self.logs

        return sequence_response

    

if __name__ == "__main__":
    pass