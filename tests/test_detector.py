"""Unit tests for Batched YOLO Detector."""
import cv2
import numpy as np
import pytest
from biketrace.core.detector import BatchedYOLODetector, Detection


def test_detector_inference():
    detector = BatchedYOLODetector(model_name="yolov8n.pt", device="cpu", conf_threshold=0.3)
    img = cv2.imread("test_motorcycle.jpg")
    assert img is not None

    # Single frame detection
    dets = detector.detect_frame(img)
    assert len(dets) > 0
    classes = [d.class_name for d in dets]
    assert "motorcycle" in classes

    # Batched detection
    batch_res = detector.detect_batch([img, img])
    assert len(batch_res) == 2
    assert len(batch_res[0]) > 0
    assert len(batch_res[1]) > 0
