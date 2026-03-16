"""
Tests for Robot_Tasks.py — RobotTasks class.

All ROS packages, hardware controllers, and OpenAI are mocked via
conftest.py so no real ROS master or API key is needed.
"""

import sys
import json
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to import RobotTasks after mocks are in place
# ---------------------------------------------------------------------------

def _get_robot_tasks_class():
    """Import RobotTasks without instantiating it (avoids rospy.init_node)."""
    import robot_tasks as Robot_Tasks
    return Robot_Tasks.RobotTasks


# ---------------------------------------------------------------------------
# possible_tasks list
# ---------------------------------------------------------------------------

EXPECTED_TASKS = {
    "navigate_to_person",
    "navigate_to_position",
    "navigate_to_object",
    "get_image_caption",
    "manipulate",
    "ask_user",
    "wait",
}


def test_possible_tasks_contains_all_7_types():
    """possible_tasks must list all 7 task type strings without instantiating."""
    # We inspect the value set in __init__ by examining the source rather than
    # constructing the class (which would call rospy.init_node and open files).
    import robot_tasks as rt_mod
    import ast
    import inspect

    src = inspect.getsource(rt_mod.RobotTasks.__init__)
    # Find the possible_tasks assignment in source
    for line in src.splitlines():
        if "possible_tasks" in line and "[" in line:
            # Extract the list literal
            start = line.index("[")
            end = line.rindex("]") + 1
            tasks = ast.literal_eval(line[start:end])
            assert set(tasks) == EXPECTED_TASKS
            return
    pytest.fail("Could not find possible_tasks assignment in RobotTasks.__init__")


# ---------------------------------------------------------------------------
# navigate_to_position — coordinate validation
# ---------------------------------------------------------------------------

def test_navigate_to_position_rejects_short_coordinate():
    """navigate_to_position must assert len(coordinate) == 7."""
    RobotTasks = _get_robot_tasks_class()

    # Patch everything __init__ touches so we can construct the object
    with patch.object(RobotTasks, "__init__", lambda self: None):
        rt = RobotTasks.__new__(RobotTasks)
        # Manually set attributes navigate_to_position uses
        rt.active_server = ""
        rt.goal_pub = MagicMock()

        with pytest.raises(AssertionError):
            rt.navigate_to_position("1.0, 2.0, 3.0")  # only 3 elements


def test_navigate_to_position_accepts_7_element_coordinate():
    """navigate_to_position should not raise for a valid 7-element coordinate."""
    RobotTasks = _get_robot_tasks_class()

    with patch.object(RobotTasks, "__init__", lambda self: None):
        rt = RobotTasks.__new__(RobotTasks)
        rt.active_server = ""
        pub_mock = MagicMock()
        rt.goal_pub = pub_mock

        coord = "1.0, 2.0, 0.0, 0.0, 0.0, 0.707, 0.707"
        # Should not raise
        rt.navigate_to_position(coord)
        assert pub_mock.publish.called


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------

def test_wait_cancels_and_clears_sequence():
    RobotTasks = _get_robot_tasks_class()

    with patch.object(RobotTasks, "__init__", lambda self: None):
        rt = RobotTasks.__new__(RobotTasks)
        rt.sequence = "some_sequence"
        rt.cancel_pub = MagicMock()
        rt.task_info_pub = MagicMock()

        rt.wait()

        rt.cancel_pub.publish.assert_called_once()
        rt.task_info_pub.publish.assert_called_once()
        assert rt.sequence == ""
