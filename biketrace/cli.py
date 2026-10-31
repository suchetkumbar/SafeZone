"""Command Line Interface for BikeTrace CCTV Stolen Bike Forensics."""
import sys
import argparse
import cv2
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console

from .config import ForensicsConfig, TargetSpec
from .pipeline import ForensicsPipeline
from .feeds.rtsp_client import RTSPProfile, test_rtsp_connection

console = Console()


def interactive_select_roi(source: str) -> Optional[Tuple[float, float, float, float]]:
    """Open an OpenCV window on the first frame allowing user to draw a bounding box around the bike."""
    console.print("[cyan]Opening video feed to select Target Parking Zone ROI...[/cyan]")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        console.print(f"[bold red]Error: Could not open source {source} to select ROI.[/bold red]")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        console.print("[bold red]Error: Could not read frame from source.[/bold red]")
        return None

    window_name = "Select Target Bike / Parking Zone ROI (Drag box, then press ENTER or SPACE; 'c' to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        console.print("[yellow]No ROI selected. Defaulting to full-scene automatic scan.[/yellow]")
        return None

    coords = (float(x), float(y), float(x + w), float(y + h))
    console.print(f"[bold green]Locked Target ROI: {coords} (x1, y1, x2, y2)[/bold green]")
    return coords


def parse_roi_str(roi_str: str) -> Tuple[float, float, float, float]:
    """Parse comma-separated 'x1,y1,x2,y2'."""
    parts = [float(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 4 numbers separated by commas: x1,y1,x2,y2")
    return (parts[0], parts[1], parts[2], parts[3])


def main():
    parser = argparse.ArgumentParser(
        description="BikeTrace: CCTV Stolen 2-Wheeler Forensics & Suspect Identification Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast scan with interactive ROI selection:
  biketrace scan path/to/cctv.mp4 --select-roi

  # Scan a folder of DVR 15-minute chunk videos:
  biketrace scan D:/CCTV_Backups/2026-10-31/ --fast

  # Scan using a reference photo of the stolen scooter:
  biketrace scan cctv_corridor.mp4 --ref-image my_activa_red.jpg

  # Scan directly from a Dahua or Hikvision live RTSP stream:
  biketrace scan "rtsp://admin:pass123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

  # Test an RTSP camera connection:
  biketrace test-rtsp "rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101"
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: scan
    scan_parser = subparsers.add_parser("scan", help="Run forensic scan on video, directory, or RTSP stream")
    scan_parser.add_argument("source", help="Path to video file, folder of DVR chunks, RTSP stream URL, or webcam index")
    scan_parser.add_argument("--roi", type=str, help="Parking zone coordinates as 'x1,y1,x2,y2'")
    scan_parser.add_argument("--select-roi", action="store_true", help="Launch interactive graphical window to draw ROI on the bike")
    scan_parser.add_argument("--ref-image", type=str, help="Path to photo of the stolen 2-wheeler for visual matching")
    scan_parser.add_argument("--category", choices=["motorcycle", "bicycle", "both"], default="both", help="Target category (default: both)")
    scan_parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint (default: yolov8n.pt for blazing speed)")
    scan_parser.add_argument("--device", default="auto", help="Compute device: 'auto', 'cuda', 'cpu'")
    scan_parser.add_argument("--stride", type=int, default=2, help="Frame stride when quiet (default: 2)")
    scan_parser.add_argument("--no-fast", action="store_true", help="Disable motion gating (scan every frame)")
    scan_parser.add_argument("--no-video", action="store_true", help="Disable generating annotated video for faster execution")
    scan_parser.add_argument("--output-dir", default="forensics_output", help="Directory to save report and snapshots")

    # Command: test-rtsp
    rtsp_parser = subparsers.add_parser("test-rtsp", help="Test connectivity to an RTSP stream")
    rtsp_parser.add_argument("url", help="RTSP stream URL")

    # Command: build-rtsp
    build_rtsp_parser = subparsers.add_parser("build-rtsp", help="Construct vendor RTSP URL for CCTV brands")
    build_rtsp_parser.add_argument("--brand", default="hikvision", choices=["hikvision", "dahua", "cpplus", "reolink", "axis", "generic_onvif"])
    build_rtsp_parser.add_argument("--ip", required=True, help="IP address of DVR/NVR/Camera")
    build_rtsp_parser.add_argument("--user", default="admin", help="Username")
    build_rtsp_parser.add_argument("--password", required=True, help="Password")
    build_rtsp_parser.add_argument("--port", type=int, default=554, help="RTSP port (default: 554)")
    build_rtsp_parser.add_argument("--channel", type=int, default=1, help="Camera channel number")
    build_rtsp_parser.add_argument("--sub-stream", action="store_true", help="Use sub-stream (lower bandwidth/fast)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "test-rtsp":
        console.print(f"[cyan]Testing RTSP stream: {args.url}...[/cyan]")
        ok = test_rtsp_connection(args.url)
        if ok:
            console.print("[bold green]✅ Connection Successful! RTSP stream is accessible and decoding frames.[/bold green]")
        else:
            console.print("[bold red]❌ Connection Failed! Verify IP, port, credentials, or firewall settings.[/bold red]")
        sys.exit(0 if ok else 1)

    if args.command == "build-rtsp":
        url = RTSPProfile.build_url(
            brand=args.brand,
            ip=args.ip,
            user=args.user,
            password=args.password,
            port=args.port,
            channel=args.channel,
            sub_stream=args.sub_stream
        )
        console.print(f"[bold green]Constructed RTSP URL:[/bold green]\n{url}")
        sys.exit(0)

    if args.command == "scan":
        target_roi = None
        if args.select_roi:
            target_roi = interactive_select_roi(args.source)
        elif args.roi:
            target_roi = parse_roi_str(args.roi)

        target_spec = TargetSpec(
            category=args.category,
            roi=target_roi,
            ref_image_path=args.ref_image,
        )

        cfg = ForensicsConfig(
            source=args.source,
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
            fast_mode=not args.no_fast,
            base_fps_stride=args.stride,
            export_annotated_video=not args.no_video,
            target=target_spec
        )

        pipeline = ForensicsPipeline(cfg)
        pipeline.run()


if __name__ == "__main__":
    main()
