# 🚨 BikeTrace AI: CCTV Stolen 2-Wheeler Forensics & Suspect Identification Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green.svg)](https://github.com/ultralytics/ultralytics)

> **High-speed computer vision forensics to parse hours of CCTV camera feeds in minutes, pinpoint when a stolen motorcycle, scooter, or bicycle departed, and extract high-resolution snapshots of the suspect who last accessed it.**

---

## ⚡ Why "Blazing Speed"?

CCTV footage typically contains 15 to 30 frames per second over 12, 24, or 48+ hours (1 to 5+ million frames). Running deep neural networks naively on every frame takes hours. **BikeTrace** solves this with a **4-tier acceleration architecture**:

1. **Two-Stage Spatial Motion Gating (`FastMotionFilter`)**:
   Uses an accelerated background subtractor with morphological filtering downsampled to thumbnail resolution. If no motion occurs in the scene or within the bike's designated parking spot, **the frame is discarded in <1 ms**, boosting scan speeds to **150–300+ FPS**.
2. **Adaptive Temporal Striding**:
   During periods of inactivity, the engine samples every $N$th frame (configurable via `--stride`). When motion or a suspect is spotted, it automatically tightens the temporal window.
3. **Batched Neural Inference (`BatchedYOLODetector`)**:
   Candidate frames are grouped into mini-batches and processed concurrently on GPU (CUDA) or multi-core CPU.
4. **Asynchronous Threaded Frame Prefetcher (`FeedReader`)**:
   Video decompression (OpenCV FFMPEG backend) runs in a background thread, preventing I/O starvation.

---

## 🚀 Quick Start (Instant Demo)

You can generate a realistic CCTV simulation scenario and run the forensics pipeline immediately:

```bash
# 1. Clone repository
git clone https://github.com/your-username/bikestolen.git
cd bikestolen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate a realistic 14-second CCTV theft scenario
python generate_test_cctv.py

# 4. Run the forensic scan
python run_forensics.py scan sample_test_cctv.mp4 --roi 380,200,680,450 --output-dir forensics_output
```

**Output:**
```
⚡ Initializing BikeTrace Forensics Engine on device: CPU
📹 Feed opened: 960x540 @ 25.0 FPS
Total Frames: 350 (~00:00:14.000)
  Scanning CCTV footage... 100%

            🚨 CCTV FORENSIC INVESTIGATION SUMMARY            
┌────────────────────────┬───────────────────────────────────┐
│ Field                  │ Forensic Result                   │
├────────────────────────┼───────────────────────────────────┤
│ Incident Status        │ STOLEN_OR_DEPARTED                │
│ Footage Duration       │ 00:00:13.920 (175 frames)         │
│ Scan Speed             │ 26.0 FPS (4.2x real-time)         │
│ Target 2-Wheeler ID    │ Track #1                          │
│ Last Stationary Time   │ 00:00:09.760                      │
│ Departure / Theft Time │ 00:00:09.840                      │
│ Primary Suspect        │ Track #4 (Confidence: 100.0%)     │
│ Suspect First Contact  │ 00:00:09.520                      │
│ Suspect Exit Contact   │ 00:00:11.920                      │
│ Interaction Duration   │ 2.4 seconds                       │
│ Observed Action        │ Interacted with bike at departure │
└────────────────────────┴───────────────────────────────────┘
```

---

## 🖥️ Interactive Web Dashboard (Streamlit GUI)

If you prefer a graphical interface rather than the command line, launch the included dashboard:

```bash
streamlit run app.py
```

### Dashboard Features:
- **Visual Video Uploader & RTSP Stream Builder**: Connect to live cameras or drag-and-drop local video files.
- **Interactive ROI Box Selection**: Adjust sliders directly on the video's first frame to lock the parking bay.
- **Photo Upload for Visual Matching**: Upload a photo of the stolen vehicle to identify it across multiple parked 2-wheelers.
- **Suspect Gallery**: View cropped images of the suspect with sharpness metrics (Laplacian focus score).
- **Interactive Timeline**: Scrub through events and view keyframe progressions.
- **One-Click Export**: Download self-contained HTML dossiers, JSON logs, and annotated evidence video.

---

## 📹 How to Integrate Existing Camera Feeds

CCTV systems export feeds in diverse formats. BikeTrace natively handles all of them:

### Form 1: Single Video Files (.mp4, .avi, .mkv, .mov, .ts, .dav, .h264)
Export footage from your DVR/NVR onto a USB drive and point directly to the file:
```bash
python run_forensics.py scan "C:/CCTV_Footage/parking_cam_20261031.mp4" --select-roi
```
> **Tip:** Use `--select-roi` to open an interactive preview window where you can click and drag a box around where the bike was parked. Press `ENTER` to confirm.

### Form 2: Partitioned DVR/NVR Folders (Chunked 15-min / 30-min Clips)
DVRs (Hikvision, Dahua, CP Plus, Axis) often split recordings into smaller segments (e.g., `ch01_20261031_080000.mp4`, `ch01_20261031_081500.mp4`).  
Simply pass the **folder path**:
```bash
python run_forensics.py scan "D:/DVR_Backups/Camera04_Oct31/" --fast --output-dir case_1031
```
BikeTrace automatically sorts files chronologically using natural ordering and evaluates them as one continuous unbroken timeline!

### Form 3: Live RTSP / RTMP Network Camera Streams
Directly monitor or ingest from IP cameras, NVRs, or DVRs over LAN/WAN:
```bash
python run_forensics.py scan "rtsp://admin:password123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
```

#### RTSP URL Reference by Camera Brand:

| Brand | Main Stream URL Template | Sub-Stream URL (Fast) |
| :--- | :--- | :--- |
| **Hikvision** | `rtsp://<user>:<pwd>@<ip>:554/Streaming/Channels/101` | `rtsp://<user>:<pwd>@<ip>:554/Streaming/Channels/102` |
| **Dahua / CP Plus**| `rtsp://<user>:<pwd>@<ip>:554/cam/realmonitor?channel=1&subtype=0` | `rtsp://<user>:<pwd>@<ip>:554/cam/realmonitor?channel=1&subtype=1` |
| **Reolink** | `rtsp://<user>:<pwd>@<ip>:554/h264Preview_01_main` | `rtsp://<user>:<pwd>@<ip>:554/h264Preview_01_sub` |
| **Axis** | `rtsp://<user>:<pwd>@<ip>:554/axis-media/media.amp` | — |
| **Generic ONVIF** | `rtsp://<user>:<pwd>@<ip>:554/onvif1` | `rtsp://<user>:<pwd>@<ip>:554/onvif2` |

> **RTSP URL Helper Tool:**  
> Use the built-in CLI helper to construct your URL:
> ```bash
> python run_forensics.py build-rtsp --brand hikvision --ip 192.168.1.50 --password mysecret --channel 2
> ```
> Test if the stream is accessible:
> ```bash
> python run_forensics.py test-rtsp "rtsp://admin:mysecret@192.168.1.50:554/Streaming/Channels/201"
> ```

### Form 4: USB Cameras / HDMI Capture Cards
To process from a connected USB webcam or hardware HDMI capture card plugged into the DVR monitor output:
```bash
# 0 is the default camera index
python run_forensics.py scan 0 --fast
```

### Form 5: Image Sequences / Timelapse Folders
If your DVR exported a directory of time-stamped JPEG/PNG frames:
```bash
python run_forensics.py scan "C:/TimelapseFrames/" --stride 1
```

---

## 🎯 Target 2-Wheeler Identification Modes

In many parking environments, several 2-wheelers are parked side-by-side. BikeTrace offers three complementary methods to specify the target:

### Method A: Parking Spot Region of Interest (`--roi` or `--select-roi`)
Ideal when you know where the bike was parked.
```bash
# Interactive selection: click and drag box on screen
python run_forensics.py scan footage.mp4 --select-roi

# Or specify absolute pixel coordinates: x1,y1,x2,y2
python run_forensics.py scan footage.mp4 --roi 350,180,620,410
```

### Method B: Visual Reference Matching (`--ref-image`)
Ideal when you have a photograph of the stolen bike (e.g., from your phone).  
BikeTrace uses a composite MobileNetV3 deep visual embedding + 3D HSV color histogram to match candidate bikes against your photo:
```bash
python run_forensics.py scan footage.mp4 --ref-image my_scooter_photo.jpg
```

### Method C: Automated Fleet Scan (No arguments required)
If you do not specify an ROI or reference image, BikeTrace monitors all 2-wheelers in view, records their individual lifecycles, and flags any bike that was present and subsequently departed or vanished.

---

## 📑 Forensic Output & Law Enforcement Hand-off

Every investigation generates an organized evidence dossier in your `--output-dir` (default: `forensics_output/`):

```
forensics_output/
├── incident_report.html         # Interactive standalone HTML dossier
├── incident_report.json         # Machine-readable audit data with coordinates & timestamps
├── incident_report.md           # Markdown investigation summary
├── forensics_annotated_scan.mp4 # Video with HUD overlays & bounding boxes
└── snapshots/
    ├── target_bike.jpg          # Cropped image of the locked 2-wheeler
    ├── primary_suspect.jpg      # Sharpest crop of the individual who took the bike
    ├── keyframe_1_parked.jpg    # Baseline: vehicle safely parked
    ├── keyframe_2_approach.jpg  # Suspect entering vehicle zone
    ├── keyframe_3_interaction.jpg # Physical contact / tampering
    ├── keyframe_4_departure.jpg # Vehicle in motion / suspect riding off
    └── keyframe_5_aftermath.jpg # Empty parking space
```

### Key Elements of the HTML Dossier (`incident_report.html`)
- **Self-Contained & Portable**: All images are embedded as Base64 strings. You can email the single `.html` file to police officers or insurance adjusters, and it opens in any browser without needing external image folders.
- **Laplacian Sharpness Scoring**: Rather than capturing a blurry motion frame, the engine evaluates each crop's focus and automatically selects the crispest frame of the suspect.
- **Incident Timeline Table**: Millisecond-accurate log of when the suspect approached, how long they interacted, and when the bike was moved.

---

## 🛠️ CLI Reference & Options

```text
usage: biketrace scan [-h] [--roi ROI] [--select-roi] [--ref-image REF_IMAGE]
                      [--category {motorcycle,bicycle,both}] [--model MODEL]
                      [--device DEVICE] [--stride STRIDE] [--no-fast]
                      [--no-video] [--output-dir OUTPUT_DIR]
                      source
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `source` | *Required* | Path to video file, DVR folder, RTSP URL, or webcam index |
| `--roi x1,y1,x2,y2` | `None` | Coordinates of parking zone (`xmin,ymin,xmax,ymax`) |
| `--select-roi` | `False` | Launches graphical window on frame 0 to draw ROI box |
| `--ref-image PATH` | `None` | Path to reference photo of the stolen bike |
| `--category` | `both` | Vehicle type: `motorcycle`, `bicycle`, or `both` |
| `--model` | `yolov8n.pt` | YOLO weights (`yolov8n.pt` for speed, `yolov8s.pt` or `yolov8m.pt` for accuracy) |
| `--device` | `auto` | Execution device: `auto`, `cuda`, or `cpu` |
| `--stride` | `2` | Frame stride during quiet periods (e.g. `2` checks every 2nd frame) |
| `--no-fast` | `False` | Disable motion gating (forces neural net inference on every frame) |
| `--no-video` | `False` | Disables annotated video export for maximum processing speed |
| `--output-dir` | `forensics_output` | Directory where reports and snapshots are saved |

---

## 🧪 Automated Testing

Run the test suite with `pytest`:
```bash
python -m pytest tests/ -v
```

All 9 unit and integration tests verify:
- Video reader parsing and natural chronological sorting
- Fast MOG2 background motion gating and ROI isolation
- Batched YOLO detection and crop extraction
- Velocity-predicted multi-object trajectory tracking
- End-to-end incident detection and report artifact generation

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
