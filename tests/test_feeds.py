"""Unit tests for CCTV feed readers."""
import cv2
import numpy as np
import pytest
from pathlib import Path
from biketrace.feeds.reader import FeedReader, format_seconds, natural_sort_key


def test_format_seconds():
    assert format_seconds(0.0) == "00:00:00.000"
    assert format_seconds(65.5) == "00:01:05.500"
    assert format_seconds(3661.123) == "01:01:01.123"


def test_natural_sort():
    files = ["ch01_10.mp4", "ch01_2.mp4", "ch01_1.mp4", "ch01_20.mp4"]
    sorted_files = sorted(files, key=natural_sort_key)
    assert sorted_files == ["ch01_1.mp4", "ch01_2.mp4", "ch01_10.mp4", "ch01_20.mp4"]


def test_feed_reader_synthetic_video(tmp_path):
    vid_file = tmp_path / "test_mini.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_file), fourcc, 10, (320, 240))
    for i in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    reader = FeedReader(str(vid_file), stride=1)
    assert reader.total_frames == 15
    assert reader.width == 320
    assert reader.height == 240

    packets = list(reader.frames())
    assert len(packets) == 15
    assert packets[0].frame_idx == 0
    assert packets[-1].frame_idx == 14
