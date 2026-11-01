"""BikeTrace CCTV Forensics Web Application (Streamlit Dashboard)."""
import os
import sys
import tempfile
import cv2
import numpy as np
import streamlit as st
from pathlib import Path

from biketrace.config import ForensicsConfig, TargetSpec
from biketrace.pipeline import ForensicsPipeline
from biketrace.feeds.reader import format_seconds
from biketrace.feeds.rtsp_client import RTSPProfile

st.set_page_config(
    page_title="BikeTrace CCTV Forensics Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #0f172a; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 0px; }
    .sub-header { color: #94a3b8; font-size: 1.1rem; margin-bottom: 25px; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .suspect-box { background: #3b0764; border-left: 5px solid #a855f7; padding: 15px; border-radius: 8px; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚨 BikeTrace CCTV Forensics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">High-Speed Video Analytics for Stolen 2-Wheelers & Suspect Identification</div>', unsafe_allow_html=True)

# Sidebar: Feed Source & Setup
with st.sidebar:
    st.header("⚙️ CCTV Feed Input")
    input_mode = st.radio("Input Source Type", ["Upload Video File", "Local File / Directory Path", "RTSP / Live Stream"])

    source_path = ""
    output_name = ""
    temp_file_path = None

    if input_mode == "Upload Video File":
        uploaded_file = st.file_uploader("Upload CCTV Footage", type=["mp4", "avi", "mkv", "mov", "ts"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name
            source_path = temp_file_path
            output_name = Path(uploaded_file.name).stem
            st.success(f"Loaded: {uploaded_file.name}")
    elif input_mode == "Local File / Directory Path":
        source_path = st.text_input("Local File Path or DVR Chunk Folder", value="data/example.mp4")
        output_name = Path(source_path).stem or Path(source_path).name
        st.caption("Supports single video or folder of segmented 15-min DVR clips.")
    else:
        st.subheader("RTSP Stream Builder")
        brand = st.selectbox("Camera Brand", ["hikvision", "dahua", "cpplus", "reolink", "axis", "generic_onvif"])
        cam_ip = st.text_input("Camera / NVR IP", value="192.168.1.100")
        cam_user = st.text_input("Username", value="admin")
        cam_pass = st.text_input("Password", type="password", value="admin123")
        cam_ch = st.number_input("Channel Number", min_value=1, max_value=64, value=1)
        sub_stream = st.checkbox("Use Sub-Stream (Faster/Lower Bandwidth)", value=False)
        
        constructed_url = RTSPProfile.build_url(
            brand=brand, ip=cam_ip, user=cam_user, password=cam_pass, channel=cam_ch, sub_stream=sub_stream
        )
        source_path = st.text_input("Live Stream URL", value=constructed_url)
        output_name = "live_camera"

    st.markdown("---")
    st.header("🎯 Target 2-Wheeler Setup")
    category = st.selectbox("Vehicle Category", ["both", "motorcycle", "bicycle"])
    ref_image_file = st.file_uploader("Optional: Reference Photo of Stolen Bike", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.header("⚡ Performance & Models")
    model_choice = st.selectbox("YOLO Model Checkpoint", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    fast_mode = st.checkbox("Enable Motion Gating (Skip static frames)", value=True)
    frame_stride = st.slider("Frame Stride (Skip N frames)", min_value=1, max_value=10, value=2)
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"])

# Main Area
if not source_path or (input_mode == "Upload Video File" and temp_file_path is None):
    st.info("👈 Please select or upload a CCTV video feed in the sidebar to begin.")
    st.stop()

# Frame preview & ROI selector
cap = cv2.VideoCapture(source_path)
first_frame = None
total_frames = 0
fps = 25.0
width, height = 1280, 720

if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ret, frame0 = cap.read()
    if ret and frame0 is not None:
        first_frame = frame0
        height, width = frame0.shape[:2]
    cap.release()

if first_frame is not None:
    st.subheader("📍 Target Parking Spot (ROI)")
    col_preview, col_roi = st.columns([2, 1])

    with col_roi:
        st.write("Specify where the 2-wheeler was parked:")
        use_roi = st.checkbox("Set Specific Parking Zone (ROI)", value=True)
        if use_roi:
            rx1 = st.slider("X Min (Left)", 0, width, int(width * 0.1))
            ry1 = st.slider("Y Min (Top)", 0, height, int(height * 0.2))
            rx2 = st.slider("X Max (Right)", rx1 + 10, width, int(width * 0.9))
            ry2 = st.slider("Y Max (Bottom)", ry1 + 10, height, int(height * 0.9))
            target_roi = (float(rx1), float(ry1), float(rx2), float(ry2))
        else:
            target_roi = None

    with col_preview:
        preview_img = first_frame.copy()
        if use_roi and target_roi:
            cv2.rectangle(preview_img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 3)
            cv2.putText(preview_img, "PARKING ZONE ROI", (int(rx1), max(25, int(ry1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rgb_preview = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, caption=f"Feed Resolution: {width}x{height} | {total_frames} frames", use_container_width=True)

# Run Investigation
st.markdown("---")
if st.button("🚀 Start Forensics Investigation", type="primary", use_container_width=True):
    ref_img_path = None
    if ref_image_file:
        t_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        t_ref.write(ref_image_file.read())
        ref_img_path = t_ref.name

    target_spec = TargetSpec(
        category=category,
        roi=target_roi if use_roi else None,
        ref_image_path=ref_img_path
    )

    out_dir = Path("output") / (output_name or Path(source_path).stem or "cctv_recording")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ForensicsConfig(
        source=source_path,
        output_dir=str(out_dir),
        model_name=model_choice,
        device=device_choice,
        fast_mode=fast_mode,
        base_fps_stride=frame_stride,
        export_annotated_video=True,
        target=target_spec
    )

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_prog(curr, total, ts):
        if total > 0:
            pct = min(1.0, curr / total)
            progress_bar.progress(pct)
            status_text.text(f"Scanning frame {curr}/{total} (Timecode: {format_seconds(ts)})...")

    pipeline = ForensicsPipeline(cfg)
    with st.spinner("Analyzing CCTV footage at blazing speed..."):
        report = pipeline.run(progress_callback=update_prog)

    progress_bar.progress(1.0)
    status_text.text("Investigation complete!")

    # Display Metrics
    st.markdown("## 📊 Investigation Findings")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", report.status)
    m2.metric("Target Vehicle", f"Track #{report.target_bike_id or 'N/A'}")
    m3.metric("Last Stationary", format_seconds(report.last_stationary_ts))
    m4.metric("Theft / Departure", format_seconds(report.departure_ts) if report.departure_ts > 0 else "N/A")

    # Suspect Profile
    if report.primary_suspect and report.primary_suspect.best_snapshot is not None:
        st.markdown("### 🚨 Primary Suspect Identified")
        scol1, scol2 = st.columns([1, 2])
        with scol1:
            rgb_suspect = cv2.cvtColor(report.primary_suspect.best_snapshot, cv2.COLOR_BGR2RGB)
            st.image(rgb_suspect, caption=f"Suspect #{report.primary_suspect.track_id} (Sharpness: {report.primary_suspect.best_sharpness:.1f})")
        with scol2:
            st.warning(f"**Action:** {report.primary_suspect.action_description}")
            st.write(f"- **Suspicion Confidence:** {report.primary_suspect.confidence_score * 100:.1f}%")
            st.write(f"- **First Approached:** {format_seconds(report.primary_suspect.first_contact_ts)}")
            st.write(f"- **Departure Contact:** {format_seconds(report.primary_suspect.last_contact_ts)}")
            st.write(f"- **Interaction Duration:** {report.primary_suspect.interaction_duration:.1f} seconds")

    # Keyframes Gallery
    st.markdown("### 🔍 Forensic Progression Keyframes")
    kcols = st.columns(5)
    kfs = [
        ("1. Parked", report.keyframe_parked),
        ("2. Approach", report.keyframe_approach),
        ("3. Interaction", report.keyframe_interaction),
        ("4. Departure", report.keyframe_departure),
        ("5. Empty Spot", report.keyframe_aftermath),
    ]
    for col, (title, img) in zip(kcols, kfs):
        with col:
            st.caption(f"**{title}**")
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.write("N/A")

    # Timeline Table
    st.markdown("### ⏱️ Chronological Incident Timeline")
    st.table(report.timeline_events)

    # Annotated Video & Downloads
    st.markdown("### 📥 Download Evidence & Reports")
    ann_vid_path = out_dir / "forensics_annotated_scan.mp4"
    if ann_vid_path.exists():
        st.video(str(ann_vid_path))

    d1, d2, d3 = st.columns(3)
    json_path = out_dir / "incident_report.json"
    html_path = out_dir / "incident_report.html"
    md_path = out_dir / "incident_report.md"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            d1.download_button("📄 Download JSON Report", f.read(), file_name="incident_report.json", mime="application/json")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            d2.download_button("🌐 Download HTML Dossier", f.read(), file_name="incident_report.html", mime="text/html")
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            d3.download_button("📝 Download Markdown Summary", f.read(), file_name="incident_report.md", mime="text/markdown")
