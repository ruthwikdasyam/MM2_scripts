"""
Shared fixtures and module-level mocks for the MM2_scripts test suite.

All ROS, OpenAI, and robot-hardware packages are stubbed out here so that
tests can import the production modules without a live ROS environment or
an OpenAI API key.  Heavy optional deps (cv2, PIL, numpy, pydantic) are
also stubbed so the suite runs in a minimal Python environment.
"""

import sys
import json
import tempfile
import os

# Add src/ to path so tests can import production modules by flat name
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import base64
import struct
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# Stub out packages that may not be installed in the test environment
# ---------------------------------------------------------------------------

# --- pydantic ---
if "pydantic" not in sys.modules:
    pydantic_mock = MagicMock()
    # BaseModel needs to be a real class so language.py can subclass it
    class _BaseModel:
        pass
    pydantic_mock.BaseModel = _BaseModel
    pydantic_mock.Field = lambda *a, **kw: None
    sys.modules["pydantic"] = pydantic_mock

# --- numpy ---
if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()

# --- cv2 ---
if "cv2" not in sys.modules:
    cv2_mock = MagicMock()
    # imencode must return (bool, bytes-like) so base64.b64encode works
    cv2_mock.imencode.return_value = (True, b"\xff\xd8\xff\xe0test_jpeg_bytes")
    cv2_mock.IMWRITE_JPEG_QUALITY = 1
    cv2_mock.INTER_AREA = 3
    sys.modules["cv2"] = cv2_mock

# --- PIL / Pillow ---
if "PIL" not in sys.modules:
    sys.modules["PIL"] = MagicMock()
    sys.modules["PIL.Image"] = MagicMock()

# --- ROS and robot-hardware packages ---
_ros_stubs = [
    "rospy",
    "actionlib",
    "actionlib_msgs",
    "actionlib_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "cv_bridge",
    "mobilegello",
    "mobilegello.gello_controller",
    "approach_object",
    "approach_object.msg",
    "tf2_ros",
    "tf2_geometry_msgs",
]
for _mod in _ros_stubs:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# --- openai ---
if "openai" not in sys.modules:
    sys.modules["openai"] = MagicMock()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_jsonl(tmp_path):
    """Return a factory that writes a list of dicts to a temp JSONL file."""
    def _make(records, filename="test.jsonl"):
        p = tmp_path / filename
        with open(p, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return str(p)
    return _make


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Replace the module-level OpenAI client in language with a MagicMock."""
    import language as lang_mod
    mock_client = MagicMock()
    monkeypatch.setattr(lang_mod, "client", mock_client)
    return mock_client
