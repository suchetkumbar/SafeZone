"""Configuration dataclasses for CCTV Stolen Bike Forensics."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


@dataclass
class TargetSpec:
    """Target 2-wheeler specification."""
    # Target category: 'motorcycle' (COCO 3), 'bicycle' (COCO 1), or 'both'
    category: str = "both"
    # ROI: (xmin, ymin, xmax, ymax) in absolute pixel coordinates or normalized [0..1]
    roi: Optional[Tuple[float, float, float, float]] = None
    # Path to reference image of the bike
    ref_image_path: Optional[str] = None
    # Optional color or description hint
    color_description: Optional[str] = None


@dataclass
class ForensicsConfig:
    """Master configuration for the video forensics pipeline."""
    # Input source: single file, directory of DVR cuts, RTSP URL, or webcam index
    source: str = ""
    # Output directory for reports, snapshots, and evidence clips
    output_dir: str = "forensics_output"
    
    # Model settings
    model_name: str = "yolov8n.pt"  # Ultra-fast default, or yolov8s.pt / yolov8m.pt
    device: str = "auto"           # 'auto', 'cuda', 'cpu'
    conf_threshold: float = 0.40   # Detection confidence
    iou_threshold: float = 0.45    # NMS IoU threshold
    batch_size: int = 8            # Inference batch size for speed
    
    # Speed & Motion Gating settings
    fast_mode: bool = True         # Enable two-stage motion gating + adaptive stride
    base_fps_stride: int = 2       # Scan every Nth frame when quiet (e.g. 2 means ~12-15 fps reduction)
    motion_gate_enabled: bool = True # Fast MOG2 background subtraction filter
    motion_threshold: float = 0.005 # Fraction of pixels that must change to trigger detection
    roi_motion_only: bool = True   # When ROI is set, check motion specifically in the ROI
    
    # Forensics & Incident Analysis
    proximity_radius_ratio: float = 0.35 # Proximity threshold as ratio of frame dimension
    interaction_frames_min: int = 3      # Minimum co-occurring frames to count as an interaction
    pre_event_seconds: float = 5.0       # Seconds of video to capture before incident
    post_event_seconds: float = 8.0      # Seconds of video to capture after incident
    
    # Export options
    export_annotated_video: bool = True
    export_evidence_clips: bool = True
    export_json: bool = True
    export_html: bool = True
    export_markdown: bool = True
    
    # Target
    target: TargetSpec = field(default_factory=TargetSpec)

    def resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' if available else 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
