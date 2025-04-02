from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List
from dataclasses import dataclass, field
import cv2
import base64
import io
from PIL import Image

client = OpenAI()

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
        self.cap = cv2.VideoCapture(2)  # Open camera at index 1

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
        if not task=="pickup" or task=="dropoff" or task=="caption":
            return print(f"Task {task} not in pickup/dropoff for Gripper")
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
                        {"type": "text", "text": f"Analyze the image and decide the correct action for the '{task}' task."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }
            ],
        )


        # Extract the response
        result = response.choices[0].message.content
        print(f"Model Response for {task}: {result}")

        # Convert response to integer for further processing
        # result = int(result)
        return result


    def get_response(self):
        
        # get query from user
        self.user_query = input("Hey, how can i help you?\n")

        # data from previous attempts
        if self.logs == "":
            add_on = ""
        else:
            add_on = f"This is the log for previous attempt and the robot was interrupted. {self.logs} So, consider this and plan again."

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are an intelligent assistant for a mobile manipulator robot.

                    ### Robot Capabilities:
                    - The **mobile robot** can navigate only to these predefined locations/people: **{self.loc_options}**. It **cannot** go anywhere else.
                    - The **manipulator** can perform these predefined tasks: **{self.arm_options}**.
                    - It also has 'wait' option, stopping everything and waiting for further instructions.
                    - The robot can also go 'home' to sleep.

                    ### Task:
                    Your goal is to generate a valid sequence of tasks for the robot based on the user's query.

                    ### Constraints:
                    1. **Strictly use only the provided names** for locations/people. Don't return person or task is **not** in the lists.
                    2. The generated sequence should alternate as follows:  
                    - **[Person Name] → [Manipulator Task] → [Person Name] → [Manipulator Task] → ...**  usually, or as user prompts
                    - Ensure the sequence follows a logical flow and provides a valid reason for each step.
                    3. Understand the query and place the pickup and dropoff modes to suit the tasks that user wants to perform with manipulation.
                    4. If the user requests an action involving an **unavailable person or task**, return:
                    - `sequence: `
                    - `reason: Reason for the failure`
                    5. The sequence can be **any length** depending on the user's request.

                    {add_on}
                    """
                },
                {
                    "role": "user",
                    "content": f"Here is the user's query: {self.user_query}."
                }
            ],
            response_format=TaskSequence, # structured response from class - TaskSequence
        )
        self.logs= f"User input: {self.user_query}. Response: {response.choices[0].message.parsed.sequence}" + self.logs

        return response
    

if __name__ == "__main__":
    pass