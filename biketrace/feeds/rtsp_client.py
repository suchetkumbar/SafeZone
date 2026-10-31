"""RTSP Stream helper and connection builder for Hikvision, Dahua, CP Plus, Axis, Reolink, etc."""
import os
import cv2
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger("biketrace.rtsp")


class RTSPProfile:
    """Standard RTSP URL formats for major CCTV DVR/NVR manufacturers."""
    
    TEMPLATES: Dict[str, str] = {
        "hikvision_main": "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        "hikvision_sub": "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}02",
        "dahua_main": "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "dahua_sub": "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
        "cpplus_main": "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "reolink_main": "rtsp://{user}:{password}@{ip}:{port}/h264Preview_{channel_2d}_main",
        "reolink_sub": "rtsp://{user}:{password}@{ip}:{port}/h264Preview_{channel_2d}_sub",
        "axis_main": "rtsp://{user}:{password}@{ip}:{port}/axis-media/media.amp",
        "generic_onvif": "rtsp://{user}:{password}@{ip}:{port}/onvif1",
    }

    @classmethod
    def build_url(
        cls,
        brand: str,
        ip: str,
        user: str = "admin",
        password: str = "admin123",
        port: int = 554,
        channel: int = 1,
        sub_stream: bool = False
    ) -> str:
        """Construct vendor-specific RTSP stream URL."""
        stream_type = "sub" if sub_stream else "main"
        key = f"{brand.lower()}_{stream_type}"
        if key not in cls.TEMPLATES:
            # Fallback to dahua/cpplus standard
            key = f"dahua_{stream_type}"

        template = cls.TEMPLATES[key]
        return template.format(
            user=user,
            password=password,
            ip=ip,
            port=port,
            channel=channel,
            channel_2d=f"{channel:02d}"
        )


def test_rtsp_connection(rtsp_url: str, timeout_sec: int = 5) -> bool:
    """Test if an RTSP camera stream is accessible."""
    # Set OpenCV FFMPEG transport flag to TCP for reliable transmission
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    cap.release()
    return bool(ret and frame is not None)
