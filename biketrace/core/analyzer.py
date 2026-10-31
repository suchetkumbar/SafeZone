"""Forensic analysis engine: target 2-wheeler identification, suspect attribution, and timeline reconstruction."""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from .detector import Detection
from .tracker import TrackedObject, FastTracker, bbox_iou
from .visual_matcher import VisualMatcher, calculate_sharpness
from ..feeds.reader import format_seconds


@dataclass
class SuspectProfile:
    """Suspect identity and forensic evidence."""
    track_id: int
    first_contact_ts: float
    last_contact_ts: float
    interaction_duration: float
    min_distance_to_bike: float
    confidence_score: float # 0.0 to 1.0 likelihood of being the actor
    best_snapshot: Optional[np.ndarray] = None
    best_sharpness: float = 0.0
    evidence_frames: List[int] = field(default_factory=list)
    action_description: str = "Interacted with target 2-wheeler"


@dataclass
class ForensicReport:
    """Complete investigation summary and evidence metadata."""
    target_bike_id: Optional[int] = None
    target_class: str = "motorcycle"
    target_initial_bbox: Optional[Tuple[int, int, int, int]] = None
    target_initial_crop: Optional[np.ndarray] = None
    
    status: str = "UNDETERMINED" # STOLEN, DEPARTED, STILL_PRESENT, NOT_FOUND
    first_seen_ts: float = 0.0
    last_stationary_ts: float = 0.0
    departure_ts: float = 0.0
    last_seen_ts: float = 0.0
    
    primary_suspect: Optional[SuspectProfile] = None
    accomplices: List[SuspectProfile] = field(default_factory=list)
    
    # Forensic keyframe images
    keyframe_parked: Optional[np.ndarray] = None
    keyframe_approach: Optional[np.ndarray] = None
    keyframe_interaction: Optional[np.ndarray] = None
    keyframe_departure: Optional[np.ndarray] = None
    keyframe_aftermath: Optional[np.ndarray] = None
    
    timeline_events: List[Dict[str, Any]] = field(default_factory=list)
    total_frames_analyzed: int = 0
    total_duration_sec: float = 0.0


class ForensicsAnalyzer:
    """
    Analyzes trajectories of 2-wheelers and persons to pinpoint theft events and suspect identity.
    """
    def __init__(
        self,
        target_roi: Optional[Tuple[float, float, float, float]] = None,
        ref_image_path: Optional[str] = None,
        target_category: str = "both",
        proximity_radius_ratio: float = 0.25,
        visual_matcher: Optional[VisualMatcher] = None
    ):
        self.target_roi = target_roi
        self.ref_image_path = ref_image_path
        self.target_category = target_category
        self.proximity_ratio = proximity_radius_ratio
        self.visual_matcher = visual_matcher

        self.ref_deep = None
        self.ref_color = None
        if self.ref_image_path and self.visual_matcher:
            ref_img = cv2.imread(self.ref_image_path)
            if ref_img is not None:
                self.ref_deep, self.ref_color = self.visual_matcher.extract_composite_features(ref_img)

        self.target_track_id: Optional[int] = None
        self.target_initial_bbox: Optional[Tuple[int, int, int, int]] = None
        self.target_initial_crop: Optional[np.ndarray] = None
        self.target_initial_pos: Optional[Tuple[float, float]] = None

        # Tracking state
        self.person_contacts: Dict[int, Dict[str, Any]] = {}
        self.timeline_events: List[Dict[str, Any]] = []
        
        self.last_stationary_ts: float = 0.0
        self.departure_ts: float = 0.0
        self.departure_frame_idx: Optional[int] = None
        self.is_departed: bool = False

        # Keyframe storage
        self.keyframe_parked: Optional[np.ndarray] = None
        self.keyframe_approach: Optional[np.ndarray] = None
        self.keyframe_interaction: Optional[np.ndarray] = None
        self.keyframe_departure: Optional[np.ndarray] = None
        self.keyframe_aftermath: Optional[np.ndarray] = None

    def _roi_overlap(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> float:
        """Calculate overlap of bbox with user-specified target ROI."""
        if self.target_roi is None:
            return 0.0
        h, w = frame_shape[:2]
        rx1, ry1, rx2, ry2 = self.target_roi
        if rx2 <= 1.0 and ry2 <= 1.0:
            rx1 = int(rx1 * w)
            ry1 = int(ry1 * h)
            rx2 = int(rx2 * w)
            ry2 = int(ry2 * h)
        roi_box = (int(rx1), int(ry1), int(rx2), int(ry2))
        return bbox_iou(bbox, roi_box)

    def _calculate_proximity(
        self,
        box_person: Tuple[int, int, int, int],
        box_bike: Tuple[int, int, int, int]
    ) -> float:
        """Compute normalized distance between person and bike bboxes."""
        px1, py1, px2, py2 = box_person
        bx1, by1, bx2, by2 = box_bike

        p_cent = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
        b_cent = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
        
        # Euclidean center distance
        dist = float(np.linalg.norm(p_cent - b_cent))
        
        # If boxes overlap or touch, distance is 0
        iou = bbox_iou(box_person, box_bike)
        if iou > 0.0:
            return 0.0
        return dist

    def process_frame_analysis(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        tracker: FastTracker
    ):
        """Analyze current frame tracks to maintain target bike state and suspect interactions."""
        h, w = frame.shape[:2]
        proximity_pixel_limit = min(w, h) * self.proximity_ratio

        # Step 1: Identify target 2-wheeler if not yet anchored
        active_2wheelers = [
            t for t in tracker.active_tracks.values()
            if t.class_name in ("motorcycle", "bicycle")
        ]

        if self.target_track_id is None and active_2wheelers:
            best_candidate = None
            best_score = -1.0

            for cand in active_2wheelers:
                score = 0.0
                # If ROI provided, prioritize ROI overlap
                if self.target_roi is not None:
                    overlap = self._roi_overlap(cand.latest_bbox, (h, w))
                    score += overlap * 2.0
                
                # If reference image provided, compute visual similarity
                if self.ref_deep is not None and cand.best_crop is not None and self.visual_matcher is not None:
                    c_deep, c_color = self.visual_matcher.extract_composite_features(cand.best_crop)
                    v_sim = self.visual_matcher.compute_similarity(self.ref_deep, self.ref_color, c_deep, c_color)
                    score += v_sim

                # If no ROI or ref image, pick first stationary 2-wheeler
                if self.target_roi is None and self.ref_deep is None:
                    score = 1.0

                if score > best_score:
                    best_score = score
                    best_candidate = cand

            if best_candidate and best_score > 0.1:
                self.target_track_id = best_candidate.track_id
                self.target_initial_bbox = best_candidate.latest_bbox
                self.target_initial_crop = best_candidate.best_crop
                self.target_initial_pos = best_candidate.trajectory[0]
                self.last_stationary_ts = timestamp
                self.keyframe_parked = frame.copy()
                self.timeline_events.append({
                    "timestamp": timestamp,
                    "time_str": format_seconds(timestamp),
                    "frame_idx": frame_idx,
                    "type": "TARGET_LOCKED",
                    "description": f"Target {best_candidate.class_name} locked (Track ID #{best_candidate.track_id})",
                })

        # Step 2: Track target bike state and displacement
        target_track = tracker.active_tracks.get(self.target_track_id) if self.target_track_id else None
        target_bbox = None

        if target_track is not None:
            target_bbox = target_track.latest_bbox
            curr_pos = np.array(target_track.trajectory[-1])
            init_pos = np.array(self.target_initial_pos) if self.target_initial_pos else curr_pos
            displacement = float(np.linalg.norm(curr_pos - init_pos))

            if displacement > 60.0 and not self.is_departed:
                # Bike started moving away!
                self.is_departed = True
                self.departure_ts = timestamp
                self.departure_frame_idx = frame_idx
                self.keyframe_departure = frame.copy()
                self.timeline_events.append({
                    "timestamp": timestamp,
                    "time_str": format_seconds(timestamp),
                    "frame_idx": frame_idx,
                    "type": "DEPARTURE_DETECTED",
                    "description": f"Target bike in motion! Displacement: {displacement:.1f}px from parking spot.",
                })
            elif not self.is_departed:
                self.last_stationary_ts = timestamp
        else:
            # Target bike was active before, but is now absent from active tracks
            if self.target_track_id is not None and not self.is_departed:
                # Disappeared from parking spot
                self.is_departed = True
                self.departure_ts = timestamp
                self.departure_frame_idx = frame_idx
                self.keyframe_aftermath = frame.copy()
                self.timeline_events.append({
                    "timestamp": timestamp,
                    "time_str": format_seconds(timestamp),
                    "frame_idx": frame_idx,
                    "type": "TARGET_DISAPPEARED",
                    "description": "Target bike has left its original parking zone and is no longer detected.",
                })

        # Use latest known target bbox or initial bbox if departed
        effective_bike_box = target_bbox or self.target_initial_bbox

        # Step 3: Monitor persons and proximity interactions
        active_persons = [t for t in tracker.active_tracks.values() if t.class_name == "person"]

        if effective_bike_box is not None:
            for person in active_persons:
                p_box = person.latest_bbox
                dist = self._calculate_proximity(p_box, effective_bike_box)

                if dist <= proximity_pixel_limit:
                    p_id = person.track_id
                    if p_id not in self.person_contacts:
                        self.person_contacts[p_id] = {
                            "track_id": p_id,
                            "first_ts": timestamp,
                            "last_ts": timestamp,
                            "frames": [frame_idx],
                            "min_dist": dist,
                            "best_crop": person.best_crop,
                            "best_sharpness": person.best_crop_sharpness,
                            "coincident_with_departure": False
                        }
                        if self.keyframe_approach is None:
                            self.keyframe_approach = frame.copy()
                        self.timeline_events.append({
                            "timestamp": timestamp,
                            "time_str": format_seconds(timestamp),
                            "frame_idx": frame_idx,
                            "type": "PERSON_APPROACHED",
                            "description": f"Person (ID #{p_id}) entered bike interaction zone (dist: {dist:.1f}px)",
                        })
                    else:
                        rec = self.person_contacts[p_id]
                        rec["last_ts"] = timestamp
                        rec["frames"].append(frame_idx)
                        if dist < rec["min_dist"]:
                            rec["min_dist"] = dist
                        if person.best_crop_sharpness > rec["best_sharpness"]:
                            rec["best_crop"] = person.best_crop
                            rec["best_sharpness"] = person.best_crop_sharpness

                    if self.keyframe_interaction is None and len(self.person_contacts[p_id]["frames"]) >= 3:
                        self.keyframe_interaction = frame.copy()

                    # Flag if person was proximate during departure
                    if self.is_departed and abs(timestamp - self.departure_ts) <= 2.5:
                        self.person_contacts[p_id]["coincident_with_departure"] = True

    def finalize_forensics(self, total_frames: int, total_duration_sec: float) -> ForensicReport:
        """Generate final forensic incident dossier with suspect identification."""
        report = ForensicReport(
            target_bike_id=self.target_track_id,
            target_initial_bbox=self.target_initial_bbox,
            target_initial_crop=self.target_initial_crop,
            last_stationary_ts=self.last_stationary_ts,
            departure_ts=self.departure_ts,
            total_frames_analyzed=total_frames,
            total_duration_sec=total_duration_sec,
            keyframe_parked=self.keyframe_parked,
            keyframe_approach=self.keyframe_approach,
            keyframe_interaction=self.keyframe_interaction,
            keyframe_departure=self.keyframe_departure,
            keyframe_aftermath=self.keyframe_aftermath,
            timeline_events=self.timeline_events,
        )

        if not self.target_track_id:
            report.status = "NOT_FOUND"
            return report

        if self.is_departed:
            report.status = "STOLEN_OR_DEPARTED"
        else:
            report.status = "STILL_PRESENT"

        # Rank all persons who had contact with the bike
        suspect_candidates: List[SuspectProfile] = []

        for p_id, rec in self.person_contacts.items():
            duration = rec["last_ts"] - rec["first_ts"]
            min_dist = rec["min_dist"]
            coincident = rec["coincident_with_departure"]
            frames_count = len(rec["frames"])

            # Filter out spurious static detections that appeared only at frame 0 and never moved
            if frames_count <= 2 and not coincident and rec["first_ts"] == 0.0:
                continue

            # Compute suspicion confidence (0.0 .. 1.0)
            score = 0.0
            if coincident:
                score += 0.60  # Present right as bike leaves
            if min_dist == 0.0:
                score += 0.25  # Directly touching or overlapping bike
            elif min_dist < 50.0:
                score += 0.15
            
            # Duration score
            if duration >= 1.0:
                score += 0.15
            score = min(1.0, max(0.05, score))

            action = "Interacted with bike at departure" if coincident else "Approached parked bike"

            suspect_candidates.append(SuspectProfile(
                track_id=p_id,
                first_contact_ts=rec["first_ts"],
                last_contact_ts=rec["last_ts"],
                interaction_duration=duration,
                min_distance_to_bike=min_dist,
                confidence_score=score,
                best_snapshot=rec["best_crop"],
                best_sharpness=rec["best_sharpness"],
                evidence_frames=rec["frames"],
                action_description=action
            ))

        # Sort candidates by confidence score and interaction recency
        suspect_candidates.sort(
            key=lambda s: (s.confidence_score, s.last_contact_ts),
            reverse=True
        )

        if suspect_candidates:
            report.primary_suspect = suspect_candidates[0]
            report.accomplices = suspect_candidates[1:]

        return report
