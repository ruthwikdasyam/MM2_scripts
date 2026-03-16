import os

# Base workspace path
WORKSPACE_PATH = os.environ.get("MM2_WORKSPACE", "/home/nvidia/catkin_ws/src/nav_assistant")

# Paths
LOCATION_MAP_PATH = os.path.join(WORKSPACE_PATH, "jsons/location_pose_map.json")
POSES_DIR = os.path.join(WORKSPACE_PATH, "poses")
EXPERIMENT_SAVE_DIR = os.path.join(WORKSPACE_PATH, "scripts/Experiments/Ex6/Images")

# Memory files
MEMORY_DIR = "memory_files"
ROBOT_LOGS_FILE = os.path.join(MEMORY_DIR, "robot_logs.jsonl")
FILTERED_EXPERIENCES_FILE = os.path.join(MEMORY_DIR, "filtered_experiences.jsonl")
RECENT_EXPERIENCES_FILE = os.path.join(MEMORY_DIR, "recent_experiences.jsonl")

# LLM settings
LLM_MODEL = os.environ.get("MM2_LLM_MODEL", "gpt-4o")
VLM_MODEL = os.environ.get("MM2_VLM_MODEL", "gpt-4o")

# Robot defaults
DEFAULT_LOC_OPTIONS = ["ruthwik", "zahir", "amisha", "kasra"]
DEFAULT_ARM_OPTIONS = ["start_pickup", "complete_pickup", "start_dropoff", "complete_dropoff"]
