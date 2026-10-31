"""Unit tests for FastTracker and Trajectory continuity."""
import pytest
from biketrace.core.detector import Detection
from biketrace.core.tracker import FastTracker, bbox_iou


def test_bbox_iou():
    boxA = (0, 0, 100, 100)
    boxB = (50, 0, 150, 100)
    iou = bbox_iou(boxA, boxB)
    assert 0.3 < iou < 0.4

    boxC = (200, 200, 300, 300)
    assert bbox_iou(boxA, boxC) == 0.0


def test_tracker_continuity():
    tracker = FastTracker(max_disappeared=5)

    # Frame 0
    d0 = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=3, class_name="motorcycle")]
    res0 = tracker.update(d0, frame_idx=0, timestamp=0.0)
    assert res0[0].track_id == 1

    # Frame 1: Slight shift
    d1 = [Detection(bbox=(105, 102, 205, 202), confidence=0.9, class_id=3, class_name="motorcycle")]
    res1 = tracker.update(d1, frame_idx=1, timestamp=0.04)
    assert res1[0].track_id == 1  # Track ID persists!
