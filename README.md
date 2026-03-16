# MM2 Scripts — LLM-Powered Robot OS for Mobile Manipulators

An open-source robot operating system that uses LLMs (GPT-4o) for high-level task planning, visual scene understanding, and memory-based reasoning. Built and tested on a mobile manipulator combining a TurtleBot base, GELLO arm, and RealSense camera — integrated with ROS1.

---

## Architecture Overview

The system is organized as a set of ROS nodes that communicate over standard topics.

- **`high_level.py`** — Main inference node: receives user queries via `/user_input`, uses the LLM to generate a plan and reasoning (Step 1), then converts that into a structured JSON task sequence (Step 2), and publishes it to `/highlevel_response` every 5 seconds while active.

- **`language.py`** — LLM interface: wraps OpenAI GPT-4o with structured outputs via Pydantic. Defines all robot actions with usage guidance, implements keyword extraction, experience filtering by keyword, and retrieval of recent experiences from JSONL logs. Also handles VLM feedback for gripper decisions and image captioning.

- **`memory.py`** — Memory/logging node: subscribes to 18 ROS topics and logs robot state approximately every 3 seconds. Each log entry captures base position (AMCL), arm state, a VLM-generated camera caption, and current task progress. LLM calls (user input, plan, reasoning, sequence) are logged separately. All data is appended to `memory_files/robot_logs.jsonl`.

- **`Robot_Tasks.py`** — Task executor: subscribes to `/highlevel_response`, parses the JSON task sequence, and dispatches each step to real ROS actions — `move_base` for navigation, GELLO arm controller for manipulation, ObjectNav action server for object-based navigation, and VLM queries for image captions.

- **`user_input.py`** — Human-in-the-loop interface: publishes user commands to `/user_input` and displays robot questions that arrive on `/askuser`.

- **`teleop.py`** — Keyboard teleoperation for the TurtleBot base.

- **`gotopoint.py`** — Navigation utility: loads the location map and publishes a goal pose to `move_base`, with a built-in arm pickup sequence for testing.

- **`LCM/`** — LCM (Lightweight Communications & Marshalling) module for pose data exchange. Contains an auto-generated `pose_t` type (x, y, theta) with encode/decode support, plus example publisher and subscriber.

- **`Experiments/`** — Experiment logs with timestamped PNG images (captured via VLM calls) and full task trace text files for six experiments (Ex1–Ex6).

- **`config.py`** — Centralized configuration: workspace paths, memory file paths, LLM model names, and robot option lists — all overridable via environment variables.

---

## Robot Capabilities

Defined in `language.py` and executed by `Robot_Tasks.py`:

- **`navigate_to_person`** — Navigate to a known person's pre-saved location. Options: `ruthwik`, `zahir`, `amisha`, `kasra`.
- **`navigate_to_position`** — Navigate to an explicit map pose `(x, y, z, qx, qy, qz, qw)` taken from memory or coordinates.
- **`navigate_to_object`** — Use the ObjectNav action server to visually locate and approach a named object.
- **`manipulate`** — Control the GELLO arm. States: `start_pickup`, `complete_pickup`, `start_dropoff`, `complete_dropoff`.
- **`get_image_caption`** — Query the VLM with the current RealSense image and a prompt; result is published to `/task_info`.
- **`ask_user`** — Display a question to the operator and pause execution until a response arrives.
- **`wait`** — Cancel all active navigation/arm goals and idle.

---

## Prerequisites

- ROS1 Noetic (Ubuntu 20.04)
- Python 3.8+
- OpenAI API key (`OPENAI_API_KEY` environment variable)
- `mobilegello` package (GELLO arm controller)
- Intel RealSense camera with `realsense2_camera` ROS driver
- `approach_object` ROS package (ObjectNav action server)
- Python packages: `openai`, `pydantic`, `opencv-python`, `Pillow`, `numpy`, `PyYAML`

---

## Setup

**1. Clone and source ROS:**
```bash
cd ~/catkin_ws/src
git clone <repo-url> nav_assistant/scripts
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

**2. Set environment variables:**
```bash
export OPENAI_API_KEY=sk-...
export MM2_WORKSPACE=/home/nvidia/catkin_ws/src/nav_assistant  # optional, this is the default
```

**3. Install Python dependencies:**
```bash
pip install openai pydantic opencv-python Pillow numpy PyYAML
```

---

## Running

Launch each node in a separate terminal (with ROS sourced):

```bash
# Memory logger
python memory.py

# High-level LLM planner
python high_level.py

# Task executor
python Robot_Tasks.py

# User input interface
python user_input.py
```

For keyboard teleoperation of the base:
```bash
python teleop.py
```

---

## Project Structure

```
MM2_scripts/
├── high_level.py           # LLM inference node (Step 1 + Step 2 planning)
├── language.py             # OpenAI/VLM interface, memory retrieval
├── memory.py               # ROS logging node -> memory_files/robot_logs.jsonl
├── Robot_Tasks.py          # Task dispatcher (navigation, arm, vision)
├── user_input.py           # Operator I/O interface
├── teleop.py               # TurtleBot keyboard teleoperation
├── gotopoint.py            # Navigation utility / arm test
├── arm_control.py          # Low-level arm control utility
├── config.py               # Centralized config (paths, models, defaults)
├── camera_control_topic.py # Camera topic publisher
├── camera_test_realsense.py# RealSense camera test
├── marylanday.py           # Maryland Day demo script
├── LCM/
│   ├── pub.py              # LCM pose publisher example
│   ├── sub.py              # LCM pose subscriber example
│   └── exlcm/
│       ├── __init__.py
│       ├── pose_t.lcm      # LCM struct definition
│       └── pose_t.py       # Auto-generated Python LCM type
├── Experiments/
│   ├── Ex1/ ... Ex6/       # Timestamped experiment logs + images
├── memory_files/
│   ├── robot_logs.jsonl    # Live robot state + LLM logs
│   ├── filtered_experiences.jsonl
│   └── recent_experiences.jsonl
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_language.py
    ├── test_lcm.py
    └── test_robot_tasks.py
```

---

## Memory Format

Each line in `robot_logs.jsonl` is a JSON object of one of two types:

**Status entry** (logged ~every 3s):
- `timestamp`, `type: "status"`
- `robot.position.base_position` — AMCL pose `[x, y, z, qx, qy, qz, qw]`
- `robot.position.arm_position` — arm joint states
- `camera_observation` — VLM caption of current camera frame
- `task_progress` — current task name, parameter, status, info

**LLM entry** (logged when a new task sequence is received):
- `timestamp`, `type: "llm"`
- `llm.user_input` — original user query
- `llm.response` — Step 1 plan text
- `llm.reasoning` — Step 1 reasoning text
- `llm.sequence` — Step 2 JSON task sequence

---

## License

Open source. See repository for details.
