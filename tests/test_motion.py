"""Unit tests for FastMotionFilter."""
import cv2
import numpy as np
import pytest
from biketrace.core.motion_filter import FastMotionFilter


def test_motion_filter_static_vs_motion():
    mf = FastMotionFilter(threshold=0.01, cooldown_frames=0)
    base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    # Feed initial 15 warmup frames
    for _ in range(15):
        mf.check_motion(base_frame)

    # Static frame should have no motion
    has_motion, ratio, _ = mf.check_motion(base_frame)
    assert not has_motion
    assert ratio < 0.01

    # Moving block in frame
    moving_frame = base_frame.copy()
    cv2.rectangle(moving_frame, (100, 100), (300, 300), (0, 0, 0), -1)
    has_motion, ratio, _ = mf.check_motion(moving_frame)
    assert has_motion
    assert ratio > 0.01


def test_motion_filter_roi():
    # ROI focused only on right side (x: 400..600, y: 100..400)
    mf = FastMotionFilter(threshold=0.01, target_roi=(400, 100, 600, 400), cooldown_frames=0)
    base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    for _ in range(15):
        mf.check_motion(base_frame)

    # Movement strictly on LEFT side (outside ROI)
    left_move = base_frame.copy()
    cv2.rectangle(left_move, (50, 50), (150, 150), (0, 0, 0), -1)
    has_motion, ratio, _ = mf.check_motion(left_move)
    assert not has_motion  # Ignored because outside ROI!
