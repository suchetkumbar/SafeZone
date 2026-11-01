"""End-to-end integration test for Stolen Bike CCTV Forensics."""
import os
import json
import pytest
from pathlib import Path
from biketrace.config import ForensicsConfig, TargetSpec
from biketrace.pipeline import ForensicsPipeline


def test_full_forensics_pipeline(tmp_path):
    video_path = "tests/videos/sample_test_cctv.mp4"
    assert os.path.exists(video_path), "sample_test_cctv.mp4 should exist"

    out_dir = tmp_path / "e2e_output"
    target_spec = TargetSpec(
        category="motorcycle",
        roi=(380, 200, 680, 450)
    )

    cfg = ForensicsConfig(
        source=video_path,
        output_dir=str(out_dir),
        model_name="yolov8n.pt",
        device="cpu",
        fast_mode=True,
        base_fps_stride=2,
        export_annotated_video=True,
        target=target_spec
    )

    pipeline = ForensicsPipeline(cfg)
    report = pipeline.run()

    # Assertions
    assert report.target_bike_id is not None
    assert report.status == "STOLEN_OR_DEPARTED"
    assert report.primary_suspect is not None
    assert report.primary_suspect.confidence_score >= 0.70
    assert report.departure_ts > 0.0

    # Verify generated artifacts
    json_path = out_dir / "incident_report.json"
    html_path = out_dir / "incident_report.html"
    md_path = out_dir / "incident_report.md"
    video_out = out_dir / "forensics_annotated_scan.mp4"

    assert json_path.exists()
    assert html_path.exists()
    assert md_path.exists()
    assert video_out.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["incident_status"] == "STOLEN_OR_DEPARTED"
        assert data["primary_suspect"]["track_id"] == report.primary_suspect.track_id
