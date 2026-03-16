"""Tests for config.py — centralized configuration module."""

import os
import importlib
import pytest


def _reload_config(monkeypatch_env=None):
    """Import (or re-import) config with an optional env override dict."""
    import config
    importlib.reload(config)
    return config


def test_all_paths_are_strings():
    import config
    assert isinstance(config.WORKSPACE_PATH, str)
    assert isinstance(config.LOCATION_MAP_PATH, str)
    assert isinstance(config.POSES_DIR, str)
    assert isinstance(config.EXPERIMENT_SAVE_DIR, str)
    assert isinstance(config.MEMORY_DIR, str)
    assert isinstance(config.ROBOT_LOGS_FILE, str)
    assert isinstance(config.FILTERED_EXPERIENCES_FILE, str)
    assert isinstance(config.RECENT_EXPERIENCES_FILE, str)
    assert isinstance(config.LLM_MODEL, str)
    assert isinstance(config.VLM_MODEL, str)


def test_default_workspace_path():
    import config
    # When MM2_WORKSPACE is not set the default should contain the nvidia path
    if "MM2_WORKSPACE" not in os.environ:
        assert config.WORKSPACE_PATH == "/home/nvidia/catkin_ws/src/nav_assistant"


def test_default_llm_model():
    import config
    if "MM2_LLM_MODEL" not in os.environ:
        assert config.LLM_MODEL == "gpt-4o"


def test_default_vlm_model():
    import config
    if "MM2_VLM_MODEL" not in os.environ:
        assert config.VLM_MODEL == "gpt-4o"


def test_default_loc_options():
    import config
    assert config.DEFAULT_LOC_OPTIONS == ["ruthwik", "zahir", "amisha", "kasra"]


def test_default_arm_options():
    import config
    assert config.DEFAULT_ARM_OPTIONS == [
        "start_pickup", "complete_pickup", "start_dropoff", "complete_dropoff"
    ]


def test_env_override_workspace(monkeypatch):
    monkeypatch.setenv("MM2_WORKSPACE", "/custom/workspace")
    import config
    importlib.reload(config)
    assert config.WORKSPACE_PATH == "/custom/workspace"
    assert config.LOCATION_MAP_PATH.startswith("/custom/workspace")
    assert config.POSES_DIR.startswith("/custom/workspace")


def test_env_override_llm_model(monkeypatch):
    monkeypatch.setenv("MM2_LLM_MODEL", "gpt-4-turbo")
    import config
    importlib.reload(config)
    assert config.LLM_MODEL == "gpt-4-turbo"


def test_env_override_vlm_model(monkeypatch):
    monkeypatch.setenv("MM2_VLM_MODEL", "gpt-4-vision-preview")
    import config
    importlib.reload(config)
    assert config.VLM_MODEL == "gpt-4-vision-preview"


def test_robot_logs_file_under_memory_dir():
    import config
    assert config.ROBOT_LOGS_FILE == os.path.join(config.MEMORY_DIR, "robot_logs.jsonl")


def test_filtered_experiences_file_under_memory_dir():
    import config
    assert config.FILTERED_EXPERIENCES_FILE == os.path.join(
        config.MEMORY_DIR, "filtered_experiences.jsonl"
    )


def test_recent_experiences_file_under_memory_dir():
    import config
    assert config.RECENT_EXPERIENCES_FILE == os.path.join(
        config.MEMORY_DIR, "recent_experiences.jsonl"
    )
