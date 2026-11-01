# BikeTrace CCTV Forensics

BikeTrace scans local CCTV recordings, segmented DVR exports, image sequences, and camera streams for 2-wheeler departures and related people. Local recordings should be placed in the project `data/` folder. Sample footage, test images, and benchmark artifacts live under `tests/`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The included `yolov8n.pt` is the default detector. The first run may require Ultralytics/PyTorch dependencies and model initialization. CPU and CUDA are supported depending on the installed PyTorch build.

## One-command CCTV scan

Copy an exported recording into `data/`, then run from the repository root:

```powershell
python run_forensics.py data/example.mp4
```

This creates `output/example/` with the HTML dossier, JSON and Markdown reports, snapshots, and (by default) an annotated MP4. The output folder name comes from the input filename without its extension. If two files share a name, choose a distinct output path with `--output-dir`.

The same direct command works with other filename extensions:

```powershell
python run_forensics.py data/parking_cam_20261101.webm
```

Optional settings can be appended to that command:

```powershell
python run_forensics.py data/example.mp4 --roi 350,180,620,410 --category motorcycle --stride 4 --device auto
```

`--roi` uses `x1,y1,x2,y2` pixel coordinates. Alternatively, add `--select-roi` to draw the parking zone interactively, or `--ref-image data/stolen-bike.jpg` to match a reference photo. For a long recording with no target area, omit these options to scan the full scene. Add `--no-video` to skip annotated-video export and reduce output size, or `--no-fast` to disable motion gating.

### Full CLI configuration

```text
python run_forensics.py <video-or-source> [options]

Options:
  --roi X1,Y1,X2,Y2          Parking-zone coordinates
  --select-roi               Select a parking zone in a preview window
  --ref-image PATH           Reference image of the target 2-wheeler
  --category {motorcycle,bicycle,both}
  --model MODEL              YOLO model file (default: yolov8n.pt)
  --device {auto,cuda,cpu}   Inference device (default: auto)
  --stride N                 Quiet-scene frame stride (default: 2)
  --no-fast                  Disable motion gating
  --no-video                 Skip annotated video export
  --output-dir PATH          Override output/<video-name>
```

The equivalent explicit subcommand remains available: `python run_forensics.py scan data/example.mp4`. To process sequential DVR chunks, pass their containing folder. RTSP and webcam sources remain supported by the explicit `scan` subcommand.

## Video format compatibility

The reader recognizes common CCTV and consumer formats including MP4, AVI, MKV, MOV, WebM, FLV, TS/MTS/M2TS, H.264/H.265 elementary streams, WMV, MPEG/MPG, 3GP, ASF, VOB, and DAV. Decoding is provided by the installed OpenCV video backend (usually FFmpeg), so a particular codec or proprietary DVR export must also be supported by that backend. If a DVR's proprietary export will not open, export/transcode it from the DVR software to MP4 or another supported container first. The example extension is `.webm` (not `.web`).

## Output layout

```text
output/
└── example/
    ├── incident_report.html
    ├── incident_report.json
    ├── incident_report.md
    ├── forensics_annotated_scan.mp4
    └── snapshots/

data/                         # Put long CCTV recordings here

tests/
├── assets/                    # Test-only images
├── videos/                    # Test and sample videos
└── results/                   # Test and benchmark reports
```

The HTML dossier is a standalone, minimal black-and-white report. Embedded CCTV frames and evidence crops remain in color.

## Test footage and suite

Generate the synthetic demonstration clip under `tests/videos/`:

```powershell
python tests/generate_test_cctv.py
```

Run tests with `python -m pytest tests/ -v`. Benchmark scripts and their generated reports are also kept under `tests/`.

## Other commands

```powershell
python run_forensics.py scan D:/DVR_Backups/2026-11-01/ --stride 4
python run_forensics.py scan 0
python run_forensics.py test-rtsp "rtsp://user:password@192.168.1.50:554/Streaming/Channels/101"
python run_forensics.py build-rtsp --brand hikvision --ip 192.168.1.50 --password secret
```

Run `python run_forensics.py --help` for command help. Distributed under the MIT License; see `LICENSE`.
