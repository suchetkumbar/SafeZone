"""YOLOv8/v11 Batched Object Detector specialized for 2-wheelers and persons."""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """Detected entity representation."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    crop: Optional[np.ndarray] = None

    @property
    def centroid(self) -> Tuple[float, float]:
        """Compute bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """Compute bounding box area."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def is_2wheeler(self) -> bool:
        """Check if detection is a motorcycle or bicycle."""
        return self.class_name in ("motorcycle", "bicycle")

    @property
    def is_person(self) -> bool:
        """Check if detection is a person."""
        return self.class_name == "person"


class BatchedYOLODetector:
    """
    High-speed YOLO detector processing single frames or batches with target class filtering.
    """
    COCO_CLASSES = {
        0: "person",
        1: "bicycle",
        3: "motorcycle"
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: str = "auto",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [0, 1, 3]  # person, bicycle, motorcycle

        # Load YOLO model
        self.model = YOLO(model_name)
        if device != "auto":
            self.model.to(device)

    def detect_batch(
        self,
        frames: List[np.ndarray],
        extract_crops: bool = True
    ) -> List[List[Detection]]:
        """
        Run batched inference on a list of frames.
        Returns a list of Detection lists, one per input frame.
        """
        if not frames:
            return []

        # Run inference
        results = self.model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False,
            device=None if self.device == "auto" else self.device
        )

        batch_detections: List[List[Detection]] = []

        for idx, res in enumerate(results):
            frame_dets: List[Detection] = []
            orig_img = frames[idx]
            h, w = orig_img.shape[:2]

            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, clss):
                    x1, y1, x2, y2 = map(int, box)
                    # Clip coordinates
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))

                    cls_name = self.COCO_CLASSES.get(cls_id, f"class_{cls_id}")
                    crop = None
                    if extract_crops and (x2 > x1) and (y2 > y1):
                        crop = orig_img[y1:y2, x1:x2].copy()

                    frame_dets.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=cls_name,
                        crop=crop
                    ))

            batch_detections.append(frame_dets)

        return batch_detections

    def detect_frame(
        self,
        frame: np.ndarray,
        extract_crops: bool = True
    ) -> List[Detection]:
        """Detect objects in a single frame."""
        res = self.detect_batch([frame], extract_crops=extract_crops)
        return res[0] if res else []
