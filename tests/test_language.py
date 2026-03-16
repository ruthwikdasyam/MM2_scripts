"""
Tests for language.py — LanguageModels class.

All OpenAI calls are mocked via conftest.py (module-level stub) and the
mock_openai_client fixture.  No real API key or ROS installation needed.
"""

import sys
import json
import base64
import tempfile
import os
from unittest.mock import MagicMock, patch
import pytest

# conftest.py stubs openai and rospy before this file runs


def _make_language():
    """Import LanguageModels fresh (conftest stubs are already in place)."""
    import language
    return language.LanguageModels()


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------

def test_init_sets_default_loc_options():
    from config import DEFAULT_LOC_OPTIONS
    lm = _make_language()
    assert lm.loc_options == DEFAULT_LOC_OPTIONS


def test_init_sets_default_arm_options():
    from config import DEFAULT_ARM_OPTIONS
    lm = _make_language()
    assert lm.arm_options == DEFAULT_ARM_OPTIONS


def test_init_accepts_custom_loc_options():
    import language
    lm = language.LanguageModels(loc_options=["alice", "bob"])
    assert lm.loc_options == ["alice", "bob"]


# ---------------------------------------------------------------------------
# robots_actions structure
# ---------------------------------------------------------------------------

EXPECTED_ACTIONS = {
    "navigate_to_person",
    "navigate_to_position",
    "navigate_to_object",
    "manipulate",
    "get_image_caption",
    "ask_user",
    "wait",
}


def test_robots_actions_has_all_7_action_types():
    lm = _make_language()
    assert set(lm.robots_actions.keys()) == EXPECTED_ACTIONS


def test_robots_actions_count():
    lm = _make_language()
    assert len(lm.robots_actions) == 7


# ---------------------------------------------------------------------------
# memory_format structure
# ---------------------------------------------------------------------------

EXPECTED_MEMORY_KEYS = {
    "base_position",
    "arm_position",
    "camera_observation",
    "task_name",
    "parameter",
    "task_status",
    "task_info",
    "user_input",
    "response",
    "reasoning",
    "sequence",
}


def test_memory_format_has_all_expected_keys():
    lm = _make_language()
    assert set(lm.memory_format.keys()) == EXPECTED_MEMORY_KEYS


# ---------------------------------------------------------------------------
# get_text_from_jsonl
# ---------------------------------------------------------------------------

def test_get_text_from_jsonl_reads_lines(tmp_path):
    p = tmp_path / "test.jsonl"
    records = [{"a": 1}, {"b": 2}, {"c": 3}]
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lm = _make_language()
    result = lm.get_text_from_jsonl(str(p))
    assert len(result) == 3
    assert json.loads(result[0]) == {"a": 1}
    assert json.loads(result[2]) == {"c": 3}


def test_get_text_from_jsonl_skips_empty_lines(tmp_path):
    p = tmp_path / "sparse.jsonl"
    p.write_text('{"x":1}\n\n{"y":2}\n')
    lm = _make_language()
    result = lm.get_text_from_jsonl(str(p))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# filter_experiences
# ---------------------------------------------------------------------------

def _status_entry(camera_obs, task_info, ts="2025-01-01 00:00:00"):
    return {
        "timestamp": ts,
        "type": "status",
        "camera_observation": camera_obs,
        "task_progress": {
            "task_name": "--",
            "parameter": "--",
            "task_status": "--",
            "task_info": task_info,
        },
    }


def test_filter_experiences_matches_keyword(tmp_path):
    input_file = str(tmp_path / "logs.jsonl")
    output_file = str(tmp_path / "filtered.jsonl")

    records = [
        _status_entry("a cup on the table", " "),
        _status_entry("person walking in hallway", " "),
        _status_entry("red cup near the sink", "cup detected"),
    ]
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lm = _make_language()
    lm.filter_experiences(input_file, output_file, ["cup"])

    with open(output_file) as f:
        results = [json.loads(l) for l in f if l.strip()]

    assert len(results) == 2
    assert all("cup" in r["camera_observation"] or "cup" in r["task_progress"]["task_info"]
               for r in results)


def test_filter_experiences_respects_100_limit(tmp_path):
    input_file = str(tmp_path / "big.jsonl")
    output_file = str(tmp_path / "out.jsonl")

    with open(input_file, "w") as f:
        for i in range(150):
            r = _status_entry(f"cup observation {i}", " ")
            f.write(json.dumps(r) + "\n")

    lm = _make_language()
    lm.filter_experiences(input_file, output_file, ["cup"])

    with open(output_file) as f:
        results = [l for l in f if l.strip()]
    assert len(results) == 100


# ---------------------------------------------------------------------------
# get_recent_20_experiences
# ---------------------------------------------------------------------------

def test_get_recent_20_experiences_returns_entries_within_window(tmp_path):
    from datetime import datetime, timedelta

    input_file = str(tmp_path / "logs.jsonl")
    output_file = str(tmp_path / "recent.jsonl")

    now = datetime.now()
    records = []
    # 5 entries within the last 3 minutes
    for i in range(5):
        ts = (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        records.append({"timestamp": ts, "type": "status",
                         "camera_observation": f"obs {i}",
                         "task_progress": {"task_name": "--", "parameter": "--",
                                           "task_status": "--", "task_info": " "}})
    # 5 entries from 10 minutes ago (outside window)
    for i in range(5):
        ts = (now - timedelta(minutes=10 + i)).strftime("%Y-%m-%d %H:%M:%S")
        records.append({"timestamp": ts, "type": "status",
                         "camera_observation": f"old obs {i}",
                         "task_progress": {"task_name": "--", "parameter": "--",
                                           "task_status": "--", "task_info": " "}})

    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lm = _make_language()
    lm.get_recent_20_experiences(input_file, output_file, time_window_minutes=5)

    with open(output_file) as f:
        results = [json.loads(l) for l in f if l.strip()]

    assert len(results) == 5
    assert all("obs" in r["camera_observation"] and "old" not in r["camera_observation"]
               for r in results)


# ---------------------------------------------------------------------------
# generate_keywords (mocked OpenAI)
# ---------------------------------------------------------------------------

def test_generate_keywords_returns_comma_separated_string(mock_openai_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "cup, table, kitchen, object, pick"
    mock_openai_client.chat.completions.create.return_value = mock_response

    lm = _make_language()
    result = lm.generate_keywords("bring me the cup from the kitchen")

    assert isinstance(result, str)
    assert "," in result
    mock_openai_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# get_encoded_image_realsense
# ---------------------------------------------------------------------------

def test_get_encoded_image_realsense_returns_base64():
    import cv2
    # cv2 is mocked in conftest; imencode returns fake JPEG bytes
    cv2.imencode.return_value = (True, b"\xff\xd8\xff\xe0fake_jpeg")

    lm = _make_language()
    # Pass a truthy non-None value as the "image"
    fake_image = object()
    result = lm.get_encoded_image_realsense(fake_image, target_size=(32, 32))

    assert result is not None
    assert isinstance(result, str)
    # Must be valid base64
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_get_encoded_image_realsense_returns_none_for_none_input():
    lm = _make_language()
    result = lm.get_encoded_image_realsense(None)
    assert result is None


# ---------------------------------------------------------------------------
# connection_check
# ---------------------------------------------------------------------------

def test_connection_check_prints_connected(capsys):
    lm = _make_language()
    lm.connection_check()
    captured = capsys.readouterr()
    assert "Connected" in captured.out
