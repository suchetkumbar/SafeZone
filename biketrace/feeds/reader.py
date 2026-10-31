"""Robust, high-throughput video feed reader supporting files, partitioned DVR directories, RTSP, and image sequences."""
import os
import re
import cv2
import glob
import time
import queue
import logging
import threading
from typing import Generator, Tuple, Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("biketrace.feeds")


def natural_sort_key(s: str):
    """Sort strings containing numbers in human/chronological order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


class FramePacket:
    """Encapsulates a frame, its metadata, global timestamp, and index."""
    __slots__ = ("frame_idx", "timestamp_sec", "formatted_time", "frame", "source_file", "is_eof")

    def __init__(
        self,
        frame_idx: int,
        timestamp_sec: float,
        formatted_time: str,
        frame: Optional[Any],
        source_file: str = "",
        is_eof: bool = False
    ):
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.formatted_time = formatted_time
        self.frame = frame
        self.source_file = source_file
        self.is_eof = is_eof


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


class FeedReader:
    """
    High-performance, multi-format CCTV feed reader.
    Handles single video files, directories of DVR split segments, RTSP streams, and webcams.
    """
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".ts", ".h264", ".h265", ".wmv", ".m4v"}
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        source: str,
        queue_size: int = 128,
        fps_override: Optional[float] = None,
        stride: int = 1
    ):
        self.source = str(source)
        self.queue_size = queue_size
        self.fps_override = fps_override
        self.stride = max(1, stride)
        
        self.files_list: List[str] = []
        self.is_stream: bool = False
        self.is_image_sequence: bool = False
        
        self._inspect_source()
        
        # Stream / Video properties
        self.fps: float = 25.0
        self.width: int = 1920
        self.height: int = 1080
        self.total_frames: int = 0
        self.estimated_duration_sec: float = 0.0
        
        self._probe_properties()
        
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _inspect_source(self):
        """Analyze source string and categorize type."""
        src_lower = self.source.lower()
        if src_lower.startswith(("rtsp://", "rtmp://", "http://", "https://")) or self.source.isdigit():
            self.is_stream = True
            return

        p = Path(self.source)
        if p.is_dir():
            # Check for videos first
            vids = []
            for ext in self.VIDEO_EXTS:
                vids.extend(p.glob(f"*{ext}"))
                vids.extend(p.glob(f"*{ext.upper()}"))
            if vids:
                vids = sorted(list(set(vids)), key=natural_sort_key)
                self.files_list = [str(v) for v in vids]
                return
            
            # Check for images
            imgs = []
            for ext in self.IMAGE_EXTS:
                imgs.extend(p.glob(f"*{ext}"))
                imgs.extend(p.glob(f"*{ext.upper()}"))
            if imgs:
                imgs = sorted(list(set(imgs)), key=natural_sort_key)
                self.files_list = [str(im) for im in imgs]
                self.is_image_sequence = True
                return
            
            raise FileNotFoundError(f"No video files or image sequences found in directory: {self.source}")

        elif p.is_file():
            if p.suffix.lower() in self.VIDEO_EXTS:
                self.files_list = [str(p)]
            elif p.suffix.lower() in self.IMAGE_EXTS:
                self.files_list = [str(p)]
                self.is_image_sequence = True
            else:
                # Attempt to open with opencv anyway
                self.files_list = [str(p)]
        else:
            raise ValueError(f"Invalid CCTV source or path does not exist: {self.source}")

    def _probe_properties(self):
        """Read metadata (fps, dimensions, frame count) from the source."""
        if self.is_stream:
            # Probe stream using VideoCapture
            cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
            if cap.isOpened():
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 25.0
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
                cap.release()
            return

        if self.is_image_sequence:
            self.fps = self.fps_override or 25.0
            self.total_frames = len(self.files_list)
            if self.files_list:
                img0 = cv2.imread(self.files_list[0])
                if img0 is not None:
                    self.height, self.width = img0.shape[:2]
            self.estimated_duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
            return

        # Multiple video files or single video file
        total_f = 0
        for idx, fpath in enumerate(self.files_list):
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                continue
            f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if idx == 0:
                self.fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
                if self.fps <= 0 or self.fps > 120:
                    self.fps = 25.0
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            total_f += f_count
            cap.release()

        self.total_frames = total_f
        self.estimated_duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

    def start(self):
        """Start background producer thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Signal reader to stop and empty queue."""
        self._stop_event.set()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def _producer_loop(self):
        """Read frames in background and populate queue."""
        global_frame_idx = 0
        global_sec = 0.0
        frame_time_step = 1.0 / self.fps if self.fps > 0 else 0.04

        if self.is_stream:
            src = int(self.source) if self.source.isdigit() else self.source
            cap = cv2.VideoCapture(src)
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Stream reconnect attempt
                    time.sleep(0.5)
                    cap.release()
                    cap = cv2.VideoCapture(src)
                    if not cap.isOpened():
                        time.sleep(1.0)
                    continue

                if global_frame_idx % self.stride == 0:
                    packet = FramePacket(
                        frame_idx=global_frame_idx,
                        timestamp_sec=global_sec,
                        formatted_time=format_seconds(global_sec),
                        frame=frame,
                        source_file=self.source
                    )
                    while not self._stop_event.is_set():
                        try:
                            self._frame_queue.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue

                global_frame_idx += 1
                global_sec += frame_time_step
            cap.release()
            self._frame_queue.put(FramePacket(-1, 0, "", None, is_eof=True))
            return

        if self.is_image_sequence:
            for idx, img_path in enumerate(self.files_list):
                if self._stop_event.is_set():
                    break
                if idx % self.stride != 0:
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                ts = idx * frame_time_step
                packet = FramePacket(
                    frame_idx=idx,
                    timestamp_sec=ts,
                    formatted_time=format_seconds(ts),
                    frame=img,
                    source_file=img_path
                )
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put(packet, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            self._frame_queue.put(FramePacket(-1, 0, "", None, is_eof=True))
            return

        # Video files
        for fpath in self.files_list:
            if self._stop_event.is_set():
                break
            cap = cv2.VideoCapture(fpath)
            local_idx = 0
            file_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            file_step = 1.0 / file_fps if file_fps > 0 else frame_time_step

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if global_frame_idx % self.stride == 0:
                    packet = FramePacket(
                        frame_idx=global_frame_idx,
                        timestamp_sec=global_sec,
                        formatted_time=format_seconds(global_sec),
                        frame=frame,
                        source_file=os.path.basename(fpath)
                    )
                    while not self._stop_event.is_set():
                        try:
                            self._frame_queue.put(packet, timeout=0.1)
                            break
                        except queue.Full:
                            continue

                global_frame_idx += 1
                global_sec += file_step
                local_idx += 1

            cap.release()

        # Mark EOF
        self._frame_queue.put(FramePacket(-1, 0, "", None, is_eof=True))

    def frames(self) -> Generator[FramePacket, None, None]:
        """Yield frame packets from the queue."""
        self.start()
        try:
            while True:
                try:
                    packet = self._frame_queue.get(timeout=2.0)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                if packet.is_eof:
                    break
                yield packet
        finally:
            self.stop()
