"""Video exporter for rendering annotated evidence footage and cut-out incident highlights."""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from ..core.detector import Detection
from ..core.analyzer import ForensicReport
from ..feeds.reader import format_seconds


class VideoEvidenceExporter:
    """Renders annotated video and extracts concise evidence clips around theft events."""
    def __init__(self, output_dir: str, fps: float = 25.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.out_path: Optional[str] = None
        self._frame_buffer: List[Tuple[float, np.ndarray]] = []

    def start_full_annotated_writer(self, filename: str, width: int, height: int) -> str:
        """Initialize VideoWriter for full run."""
        self.out_path = str(self.output_dir / filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (width, height))
        return self.out_path

    def draw_forensic_overlay(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp: float,
        target_bike_id: Optional[int] = None,
        suspect_track_id: Optional[int] = None,
        target_roi: Optional[Tuple[float, float, float, float]] = None,
        status_text: str = ""
    ) -> np.ndarray:
        """
        Render tactical forensic HUD overlays on a frame:
        - Target bike: Glowing GREEN box with target label
        - Suspect: RED box with warning label
        - Other bikes: CYAN box
        - Other pedestrians: GRAY / SLATE box
        - Parking ROI: Dashed / semi-transparent box
        - Header HUD: Timecode, status badge, FPS
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw Target ROI if specified
        if target_roi is not None:
            rx1, ry1, rx2, ry2 = target_roi
            if rx2 <= 1.0 and ry2 <= 1.0:
                rx1 = int(rx1 * w)
                ry1 = int(ry1 * h)
                rx2 = int(rx2 * w)
                ry2 = int(ry2 * h)
            cv2.rectangle(vis, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 200, 0), 2)
            cv2.putText(
                vis, "TARGET PARKING ZONE", (int(rx1), max(20, int(ry1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA
            )

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_target = (det.track_id is not None and det.track_id == target_bike_id)
            is_suspect = (det.track_id is not None and det.track_id == suspect_track_id)

            if is_target:
                color = (0, 255, 0)  # Bright Green
                label = f"TARGET BIKE #{det.track_id} ({det.confidence*100:.0f}%)"
                thickness = 3
            elif is_suspect:
                color = (0, 0, 255)  # Bright Red
                label = f"SUSPECT #{det.track_id} ({det.confidence*100:.0f}%)"
                thickness = 3
            elif det.is_2wheeler:
                color = (255, 255, 0)  # Cyan
                label = f"{det.class_name.upper()} #{det.track_id or '?'}"
                thickness = 2
            else:
                color = (180, 180, 180)  # Gray
                label = f"person #{det.track_id or '?'}"
                thickness = 1

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Label background banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 3, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        # Header Telemetry HUD Bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

        hud_text = f"CCTV FORENSICS | TIMECODE: {format_seconds(timestamp)}"
        if status_text:
            hud_text += f" | STATUS: {status_text}"
        cv2.putText(vis, hud_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def write_frame(self, frame: np.ndarray):
        """Append frame to output video."""
        if self.writer is not None:
            self.writer.write(frame)

    def close(self):
        """Release full video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def export_highlight_clip(
        self,
        source_video: str,
        start_ts: float,
        end_ts: float,
        output_name: str = "incident_highlight.mp4"
    ) -> Optional[str]:
        """Cut a high-fidelity highlight video around the incident."""
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        out_path = str(self.output_dir / output_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        start_frame = int(start_ts * fps)
        end_frame = int(end_ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        curr_frame = start_frame
        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)
            curr_frame += 1

        cap.release()
        writer.release()
        return out_path
