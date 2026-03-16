"""
Tests for LCM/exlcm/pose_t.py — the auto-generated LCM type.

No lcm runtime package needed: pose_t only uses stdlib (struct, io).
"""

import sys
import os

# Make the LCM package importable when tests are run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LCM"))

from exlcm.pose_t import pose_t


def test_pose_t_instantiates_with_defaults():
    p = pose_t()
    assert p.x == 0.0
    assert p.y == 0.0
    assert p.theta == 0.0


def test_pose_t_encode_returns_bytes():
    p = pose_t()
    encoded = p.encode()
    assert isinstance(encoded, bytes)
    # fingerprint (8 bytes) + 3 doubles (24 bytes) = 32 bytes
    assert len(encoded) == 32


def test_pose_t_encode_decode_roundtrip():
    import math
    p = pose_t()
    p.x = 1.5
    p.y = -2.75
    p.theta = 0.314

    encoded = p.encode()
    decoded = pose_t.decode(encoded)

    assert math.isclose(decoded.x, 1.5)
    assert math.isclose(decoded.y, -2.75)
    assert math.isclose(decoded.theta, 0.314)


def test_pose_t_roundtrip_various_values():
    import math
    cases = [
        (0.0, 0.0, 0.0),
        (100.0, 200.0, 3.14159),
        (-50.5, 0.001, -1.5707),
        (1e-10, 1e10, 0.0),
    ]
    for x, y, theta in cases:
        p = pose_t()
        p.x = x
        p.y = y
        p.theta = theta

        decoded = pose_t.decode(p.encode())
        assert math.isclose(decoded.x, x, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(decoded.y, y, rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(decoded.theta, theta, rel_tol=1e-9, abs_tol=1e-12)


def test_pose_t_decode_invalid_data_raises():
    import pytest
    with pytest.raises((ValueError, Exception)):
        pose_t.decode(b"\x00" * 32)


def test_pose_t_get_hash_returns_int():
    p = pose_t()
    h = p.get_hash()
    assert isinstance(h, int)


def test_pose_t_slots():
    p = pose_t()
    assert set(p.__slots__) == {"x", "y", "theta"}
