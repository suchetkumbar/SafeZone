"""End-to-End High-Speed CCTV Forensics Pipeline Orchestrator."""
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .config import ForensicsConfig
from .feeds.reader import FeedReader, FramePacket, format_seconds
from .core.motion_filter import FastMotionFilter
from .core.detector import BatchedYOLODetector
from .core.tracker import FastTracker
from .core.visual_matcher import VisualMatcher
from .core.analyzer import ForensicsAnalyzer, ForensicReport
from .reporting.report_generator import ReportGenerator
from .reporting.video_exporter import VideoEvidenceExporter

console = Console()


class ForensicsPipeline:
    """
    High-performance pipeline coordinating multi-format CCTV feeds,
    motion gating, batched inference, object tracking, and suspect forensics.
    """
    def __init__(self, config: ForensicsConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = config.resolve_device()
        console.print(f"[bold cyan]⚡ Initializing BikeTrace Forensics Engine on device: [green]{device.upper()}[/green][/bold cyan]")

        # Components
        self.motion_filter = FastMotionFilter(
            threshold=config.motion_threshold,
            target_roi=config.target.roi if config.roi_motion_only else None
        ) if config.motion_gate_enabled else None

        self.visual_matcher = VisualMatcher(device="cpu") if config.target.ref_image_path else None

        self.detector = BatchedYOLODetector(
            model_name=config.model_name,
            device=device,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            target_classes=[0, 1, 3]  # person, bicycle, motorcycle
        )

        self.tracker = FastTracker(max_disappeared=15)

        self.analyzer = ForensicsAnalyzer(
            target_roi=config.target.roi,
            ref_image_path=config.target.ref_image_path,
            target_category=config.target.category,
            proximity_radius_ratio=config.proximity_radius_ratio,
            visual_matcher=self.visual_matcher
        )

        self.report_generator = ReportGenerator(str(self.output_dir))
        self.video_exporter = VideoEvidenceExporter(str(self.output_dir))

    def run(self, progress_callback: Optional[Callable[[int, int, float], None]] = None) -> ForensicReport:
        """
        Execute forensic scan on the configured video feed.
        """
        start_wall_time = time.time()
        reader = FeedReader(
            source=self.config.source,
            stride=self.config.base_fps_stride if self.config.fast_mode else 1
        )

        total_frames = reader.total_frames
        fps = reader.fps
        width = reader.width
        height = reader.height

        self.video_exporter.fps = fps / (self.config.base_fps_stride if self.config.fast_mode else 1)
        if self.config.export_annotated_video:
            self.video_exporter.start_full_annotated_writer(
                "forensics_annotated_scan.mp4", width, height
            )

        processed_frames = 0
        motion_frames_count = 0
        skipped_static_frames = 0
        last_timestamp = 0.0

        # Batching structures
        pending_packets: List[FramePacket] = []

        console.print(f"[bold green]📹 Feed opened: {reader.width}x{reader.height} @ {reader.fps:.1f} FPS[/bold green]")
        if total_frames > 0:
            console.print(f"[cyan]Total Frames: {total_frames} (~{format_seconds(reader.estimated_duration_sec)})[/cyan]")
        else:
            console.print("[cyan]Live Stream / Dynamic Feed[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as pbar:
            task_id = pbar.add_task("[yellow]Scanning CCTV footage...", total=total_frames if total_frames > 0 else None)

            def flush_batch(packets: List[FramePacket]):
                nonlocal processed_frames, motion_frames_count
                if not packets:
                    return

                frames_img = [p.frame for p in packets]
                batch_detections = self.detector.detect_batch(frames_img, extract_crops=True)

                for p, dets in zip(packets, batch_detections):
                    processed_frames += 1
                    motion_frames_count += 1
                    
                    # Update tracker
                    tracked_dets = self.tracker.update(dets, p.frame_idx, p.timestamp_sec)
                    
                    # Run forensic analysis
                    self.analyzer.process_frame_analysis(p.frame, p.frame_idx, p.timestamp_sec, self.tracker)

                    # Export annotated frame
                    if self.config.export_annotated_video:
                        status_str = f"Target: #{self.analyzer.target_track_id or 'Searching'}"
                        if self.analyzer.is_departed:
                            status_str += " | ⚠️ VEHICLE MOVED/GONE"
                        overlay_frame = self.video_exporter.draw_forensic_overlay(
                            p.frame,
                            tracked_dets,
                            p.timestamp_sec,
                            target_bike_id=self.analyzer.target_track_id,
                            suspect_track_id=self.analyzer.primary_suspect.track_id if hasattr(self.analyzer, 'primary_suspect') and self.analyzer.primary_suspect else None,
                            target_roi=self.config.target.roi,
                            status_text=status_str
                        )
                        self.video_exporter.write_frame(overlay_frame)

            for packet in reader.frames():
                frame = packet.frame
                last_timestamp = packet.timestamp_sec

                # Update progress
                pbar.update(task_id, advance=self.config.base_fps_stride if self.config.fast_mode else 1)
                if progress_callback and total_frames > 0:
                    progress_callback(packet.frame_idx, total_frames, packet.timestamp_sec)

                # Motion Filter Stage
                has_motion = True
                if self.motion_filter is not None:
                    has_motion, _, _ = self.motion_filter.check_motion(frame)

                if not has_motion and self.config.fast_mode:
                    skipped_static_frames += 1
                    continue

                pending_packets.append(packet)

                if len(pending_packets) >= self.config.batch_size:
                    flush_batch(pending_packets)
                    pending_packets.clear()

            # Flush remaining
            if pending_packets:
                flush_batch(pending_packets)
                pending_packets.clear()

        # Close video writer
        self.video_exporter.close()

        total_elapsed = time.time() - start_wall_time
        effective_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0
        throughput_ratio = (last_timestamp / total_elapsed) if total_elapsed > 0 else 0

        # Finalize report
        report = self.analyzer.finalize_forensics(
            total_frames=processed_frames + skipped_static_frames,
            total_duration_sec=last_timestamp
        )

        # Generate reports and save assets
        report_files = self.report_generator.generate(report)

        # Print executive summary table
        self._print_terminal_summary(report, total_elapsed, effective_fps, throughput_ratio, report_files)

        return report

    def _print_terminal_summary(
        self,
        report: ForensicReport,
        elapsed_sec: float,
        fps_speed: float,
        throughput_ratio: float,
        report_files: Dict[str, str]
    ):
        """Display color-coded forensic summary in terminal."""
        table = Table(title="🚨 CCTV FORENSIC INVESTIGATION SUMMARY", style="bold red")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Forensic Result", style="white")

        table.add_row("Incident Status", f"[bold {'red' if report.status != 'STILL_PRESENT' else 'green'}]{report.status}[/]")
        table.add_row("Footage Duration", f"{format_seconds(report.total_duration_sec)} ({report.total_frames_analyzed} frames)")
        table.add_row("Scan Speed", f"{fps_speed:.1f} FPS ({throughput_ratio:.1f}x real-time in {elapsed_sec:.1f}s)")
        table.add_row("Target 2-Wheeler ID", f"Track #{report.target_bike_id or 'None'}")
        table.add_row("Last Stationary Time", format_seconds(report.last_stationary_ts))
        table.add_row("Departure / Theft Time", f"[bold red]{format_seconds(report.departure_ts)}[/]" if report.departure_ts > 0 else "N/A (Still in Spot)")

        if report.primary_suspect:
            p = report.primary_suspect
            table.add_row("Primary Suspect", f"[bold red]Track #{p.track_id}[/] (Confidence: {p.confidence_score * 100:.1f}%)")
            table.add_row("Suspect First Contact", format_seconds(p.first_contact_ts))
            table.add_row("Suspect Exit Contact", format_seconds(p.last_contact_ts))
            table.add_row("Interaction Duration", f"{p.interaction_duration:.1f} seconds")
            table.add_row("Observed Action", p.action_description)

        console.print(table)

        console.print(f"[bold green]✅ Forensic reports generated successfully in: [underline]{self.output_dir}[/underline][/bold green]")
        for fmt, path in report_files.items():
            console.print(f"  • [yellow]{fmt.upper()}:[/yellow] {path}")
