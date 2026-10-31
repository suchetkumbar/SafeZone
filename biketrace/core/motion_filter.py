"""Ultra-fast motion detection filter to skip non-event frames at hundreds of FPS."""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class FastMotionFilter:
    """
    Two-stage motion detector using MOG2 background subtraction and ROI mask filtering.
    Enables CCTV processing speeds of 150-300+ FPS by skipping static frames.
    """
    def __init__(
        self,
        threshold: float = 0.005,
        target_roi: Optional[Tuple[float, float, float, float]] = None,
        downsample_width: int = 320,
        downsample_height: int = 180,
        history: int = 200,
        var_threshold: float = 24.0,
        cooldown_frames: int = 5
    ):
        """
        Args:
            threshold: Minimum fraction of pixels changed (0.005 = 0.5%)
            target_roi: Optional (x1, y1, x2, y2) normalized [0..1] or absolute pixel coords
            downsample_width: Fast processing resolution width
            downsample_height: Fast processing resolution height
            history: MOG2 history frames
            var_threshold: Variance threshold for pixel classification
            cooldown_frames: Extra frames to process after motion ceases
        """
        self.threshold = threshold
        self.target_roi = target_roi
        self.ds_w = downsample_width
        self.ds_h = downsample_height
        self.cooldown_frames = cooldown_frames
        self._cooldown_counter = 0

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._frame_count = 0

    def check_motion(self, frame: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Check if frame contains significant motion.
        
        Returns:
            (has_motion: bool, motion_ratio: float, fg_mask: np.ndarray)
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        # Fast resize for background subtraction
        small = cv2.resize(frame, (self.ds_w, self.ds_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg_mask = self.subtractor.apply(gray)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        # If ROI is specified, focus motion evaluation in that zone
        if self.target_roi is not None:
            rx1, ry1, rx2, ry2 = self.target_roi
            # Convert normalized to downsample dimensions if needed
            if rx2 <= 1.0 and ry2 <= 1.0:
                ix1 = int(rx1 * self.ds_w)
                iy1 = int(ry1 * self.ds_h)
                ix2 = int(rx2 * self.ds_w)
                iy2 = int(ry2 * self.ds_h)
            else:
                scale_x = self.ds_w / w
                scale_y = self.ds_h / h
                ix1 = int(rx1 * scale_x)
                iy1 = int(ry1 * scale_y)
                ix2 = int(rx2 * scale_x)
                iy2 = int(ry2 * scale_y)

            # Add 20% margin around ROI to catch approaching suspects
            pad_x = int((ix2 - ix1) * 0.25)
            pad_y = int((iy2 - iy1) * 0.25)
            ix1 = max(0, ix1 - pad_x)
            iy1 = max(0, iy1 - pad_y)
            ix2 = min(self.ds_w, ix2 + pad_x)
            iy2 = min(self.ds_h, iy2 + pad_y)

            eval_mask = fg_mask[iy1:iy2, ix1:ix2]
            total_pixels = max(1, (ix2 - ix1) * (iy2 - iy1))
        else:
            eval_mask = fg_mask
            total_pixels = self.ds_w * self.ds_h

        motion_pixels = cv2.countNonZero(eval_mask)
        motion_ratio = motion_pixels / total_pixels

        # Allow initial warmup frames to build background model
        if self._frame_count < 10:
            return True, motion_ratio, fg_mask

        is_moving = motion_ratio >= self.threshold

        if is_moving:
            self._cooldown_counter = self.cooldown_frames
            return True, motion_ratio, fg_mask

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return True, motion_ratio, fg_mask

        return False, motion_ratio, fg_mask
