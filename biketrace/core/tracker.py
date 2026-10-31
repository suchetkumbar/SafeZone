"""Multi-object tracker with spatial continuity and stationary status estimation."""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .detector import Detection
from .visual_matcher import calculate_sharpness


def bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class TrackedObject:
    """Represents a continuous trajectory of a tracked object."""
    def __init__(self, track_id: int, initial_detection: Detection, frame_idx: int, timestamp: float):
        self.track_id = track_id
        self.class_name = initial_detection.class_name
        self.class_id = initial_detection.class_id
        self.latest_bbox = initial_detection.bbox
        self.latest_confidence = initial_detection.confidence
        
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        
        self.trajectory: List[Tuple[float, float]] = [initial_detection.centroid]
        self.bboxes: List[Tuple[int, int, int, int]] = [initial_detection.bbox]
        self.timestamps: List[float] = [timestamp]
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        
        self.disappeared_frames = 0
        self.best_crop = initial_detection.crop
        self.best_crop_sharpness = calculate_sharpness(initial_detection.crop)

    def predict_next_centroid(self) -> Tuple[float, float]:
        """Estimate next expected position using recent velocity."""
        curr = self.trajectory[-1]
        return (curr[0] + self.velocity[0], curr[1] + self.velocity[1])

    def update(self, detection: Detection, frame_idx: int, timestamp: float):
        """Update track with new detection."""
        self.latest_bbox = detection.bbox
        self.latest_confidence = detection.confidence
        self.last_frame = frame_idx
        self.last_timestamp = timestamp
        self.disappeared_frames = 0
        
        prev_cent = self.trajectory[-1]
        curr_cent = detection.centroid
        dt = max(1, frame_idx - self.last_frame)
        self.velocity = ((curr_cent[0] - prev_cent[0]) / dt, (curr_cent[1] - prev_cent[1]) / dt)
        
        self.trajectory.append(curr_cent)
        self.bboxes.append(detection.bbox)
        self.timestamps.append(timestamp)

        if detection.crop is not None:
            sharpness = calculate_sharpness(detection.crop)
            if sharpness > self.best_crop_sharpness:
                self.best_crop = detection.crop
                self.best_crop_sharpness = sharpness

    def mark_missed(self):
        """Increment missed frame count."""
        self.disappeared_frames += 1

    @property
    def total_displacement(self) -> float:
        """Euclidean distance from start to current position."""
        if len(self.trajectory) < 2:
            return 0.0
        start = np.array(self.trajectory[0])
        curr = np.array(self.trajectory[-1])
        return float(np.linalg.norm(curr - start))

    @property
    def is_stationary(self) -> bool:
        """Heuristic for stationary state (displacement < 35 pixels over track)."""
        return self.total_displacement < 35.0


class FastTracker:
    """High-speed IoU and velocity-aware proximity association tracker."""
    def __init__(self, max_disappeared: int = 25, iou_threshold: float = 0.15, max_centroid_dist: float = 160.0):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_centroid_dist = max_centroid_dist
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.closed_tracks: List[TrackedObject] = []

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float
    ) -> List[Detection]:
        """
        Update tracker with detections and assign track_ids.
        """
        # If no active tracks, initialize all
        if not self.active_tracks:
            for det in detections:
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.active_tracks[tid] = TrackedObject(tid, det, frame_idx, timestamp)
            return detections

        track_ids = list(self.active_tracks.keys())
        det_indices = list(range(len(detections)))

        matched_tracks = set()
        matched_dets = set()

        # Step 1: Match by IoU
        for tid in track_ids:
            track = self.active_tracks[tid]
            best_iou = self.iou_threshold
            best_det_idx = None

            for d_idx in det_indices:
                if d_idx in matched_dets:
                    continue
                det = detections[d_idx]
                if det.class_name != track.class_name:
                    continue
                iou = bbox_iou(track.latest_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_det_idx is not None:
                matched_tracks.add(tid)
                matched_dets.add(best_det_idx)
                detections[best_det_idx].track_id = tid
                track.update(detections[best_det_idx], frame_idx, timestamp)

        # Step 2: Match remaining by velocity-predicted centroid distance
        unmatched_tracks = [t for t in track_ids if t not in matched_tracks]
        for tid in unmatched_tracks:
            track = self.active_tracks[tid]
            predicted_cent = np.array(track.predict_next_centroid())
            best_dist = self.max_centroid_dist
            best_det_idx = None

            for d_idx in det_indices:
                if d_idx in matched_dets:
                    continue
                det = detections[d_idx]
                if det.class_name != track.class_name:
                    continue
                cent = np.array(det.centroid)
                dist = np.linalg.norm(predicted_cent - cent)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = d_idx

            if best_det_idx is not None:
                matched_tracks.add(tid)
                matched_dets.add(best_det_idx)
                detections[best_det_idx].track_id = tid
                track.update(detections[best_det_idx], frame_idx, timestamp)

        # Step 3: Handle unmatched active tracks
        still_unmatched = [t for t in track_ids if t not in matched_tracks]
        for tid in still_unmatched:
            track = self.active_tracks[tid]
            track.mark_missed()
            if track.disappeared_frames > self.max_disappeared:
                self.closed_tracks.append(track)
                del self.active_tracks[tid]

        # Step 4: Create new tracks for unmatched detections
        for d_idx in det_indices:
            if d_idx not in matched_dets:
                det = detections[d_idx]
                tid = self.next_track_id
                self.next_track_id += 1
                det.track_id = tid
                self.active_tracks[tid] = TrackedObject(tid, det, frame_idx, timestamp)

        return detections
