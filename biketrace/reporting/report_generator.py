"""Generates comprehensive investigative reports in JSON, HTML, and Markdown with embedded evidence."""
import os
import json
import base64
import html
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from ..core.analyzer import ForensicReport, SuspectProfile
from ..feeds.reader import format_seconds


def image_to_base64(img: Optional[np.ndarray], quality: int = 90) -> str:
    """Encode OpenCV BGR image as base64 JPEG string."""
    if img is None or img.size == 0:
        return ""
    success, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


class ReportGenerator:
    """Creates structured investigative dossiers for law enforcement and owners."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    def save_image(self, img: Optional[np.ndarray], filename: str) -> Optional[str]:
        """Save image to disk if not None."""
        if img is None or img.size == 0:
            return None
        dest = self.snapshots_dir / filename
        cv2.imwrite(str(dest), img)
        return str(dest)

    def generate(self, report: ForensicReport) -> Dict[str, str]:
        """Generate all report artifacts and return their file paths."""
        generated_files = {}

        # 1. Save all snapshot images to disk
        self.save_image(report.target_initial_crop, "target_bike.jpg")
        if report.primary_suspect and report.primary_suspect.best_snapshot is not None:
            self.save_image(report.primary_suspect.best_snapshot, "primary_suspect.jpg")

        for idx, acc in enumerate(report.accomplices):
            if acc.best_snapshot is not None:
                self.save_image(acc.best_snapshot, f"person_contact_{idx+1}.jpg")

        self.save_image(report.keyframe_parked, "keyframe_1_parked.jpg")
        self.save_image(report.keyframe_approach, "keyframe_2_approach.jpg")
        self.save_image(report.keyframe_interaction, "keyframe_3_interaction.jpg")
        self.save_image(report.keyframe_departure, "keyframe_4_departure.jpg")
        self.save_image(report.keyframe_aftermath, "keyframe_5_aftermath.jpg")

        # 2. Generate JSON report
        json_path = self.output_dir / "incident_report.json"
        json_data = self._to_dict(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        generated_files["json"] = str(json_path)

        # 3. Generate Markdown report
        md_path = self.output_dir / "incident_report.md"
        md_content = self._to_markdown(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        generated_files["markdown"] = str(md_path)

        # 4. Generate self-contained standalone HTML report
        html_path = self.output_dir / "incident_report.html"
        html_content = self._to_html(report)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        generated_files["html"] = str(html_path)

        return generated_files

    def _to_dict(self, report: ForensicReport) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        data = {
            "incident_status": report.status,
            "target_bike": {
                "track_id": report.target_bike_id,
                "initial_bounding_box": report.target_initial_bbox,
                "last_stationary_timestamp": report.last_stationary_ts,
                "last_stationary_time_formatted": format_seconds(report.last_stationary_ts),
                "departure_timestamp": report.departure_ts,
                "departure_time_formatted": format_seconds(report.departure_ts) if report.departure_ts > 0 else "N/A",
            },
            "investigation_metrics": {
                "total_frames_analyzed": report.total_frames_analyzed,
                "total_duration_seconds": report.total_duration_sec,
                "duration_formatted": format_seconds(report.total_duration_sec),
            },
            "primary_suspect": None,
            "associated_persons": [],
            "timeline": report.timeline_events
        }

        if report.primary_suspect:
            data["primary_suspect"] = {
                "track_id": report.primary_suspect.track_id,
                "first_contact_ts": report.primary_suspect.first_contact_ts,
                "first_contact_formatted": format_seconds(report.primary_suspect.first_contact_ts),
                "last_contact_ts": report.primary_suspect.last_contact_ts,
                "last_contact_formatted": format_seconds(report.primary_suspect.last_contact_ts),
                "interaction_duration_seconds": report.primary_suspect.interaction_duration,
                "suspicion_confidence_score": round(report.primary_suspect.confidence_score, 3),
                "action": report.primary_suspect.action_description
            }

        for p in report.accomplices:
            data["associated_persons"].append({
                "track_id": p.track_id,
                "first_contact_ts": p.first_contact_ts,
                "last_contact_ts": p.last_contact_ts,
                "suspicion_confidence_score": round(p.confidence_score, 3),
                "action": p.action_description
            })

        return data

    def _to_markdown(self, report: ForensicReport) -> str:
        """Render report in GitHub Flavored Markdown."""
        lines = [
            "# 🚨 CCTV FORENSIC INCIDENT REPORT: STOLEN 2-WHEELER INVESTIGATION",
            "",
            f"**Incident Status:** `{report.status}`  ",
            f"**Total Footage Processed:** `{format_seconds(report.total_duration_sec)}` ({report.total_frames_analyzed} frames)  ",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Target 2-Wheeler ID:** `#{report.target_bike_id}`",
            f"- **Last Verified Stationary At:** `{format_seconds(report.last_stationary_ts)}`",
            f"- **Exact Departure / Theft Timestamp:** `{format_seconds(report.departure_ts) if report.departure_ts > 0 else 'STILL PRESENT'}`",
            "",
        ]

        if report.primary_suspect:
            lines.extend([
                "## 2. Suspect Identification",
                "",
                f"> **PRIMARY SUSPECT (Track #{report.primary_suspect.track_id})**",
                f"- **Suspicion Confidence:** `{report.primary_suspect.confidence_score * 100:.1f}%`",
                f"- **Initial Contact:** `{format_seconds(report.primary_suspect.first_contact_ts)}`",
                f"- **Departure Exit Contact:** `{format_seconds(report.primary_suspect.last_contact_ts)}`",
                f"- **Interaction Duration:** `{report.primary_suspect.interaction_duration:.1f}s`",
                f"- **Observed Activity:** {report.primary_suspect.action_description}",
                f"- **Snapshot Saved To:** `snapshots/primary_suspect.jpg`",
                "",
            ])
        else:
            lines.extend([
                "## 2. Suspect Identification",
                "",
                "No human interaction was confirmed within proximity threshold during departure.",
                "",
            ])

        lines.extend([
            "## 3. Timeline of Events",
            "",
            "| Timecode | Event Type | Description |",
            "| :--- | :--- | :--- |",
        ])

        for ev in report.timeline_events:
            lines.append(f"| `{ev.get('time_str', 'N/A')}` | **{ev.get('type', '')}** | {ev.get('description', '')} |")

        lines.append("")
        lines.append("---")
        lines.append("*Generated by BikeTrace CCTV Forensics Toolkit*")
        return "\n".join(lines)

    def _to_html(self, report: ForensicReport) -> str:
        """Render modern, responsive standalone HTML dossier with embedded base64 images."""
        b64_bike = image_to_base64(report.target_initial_crop)
        b64_suspect = image_to_base64(report.primary_suspect.best_snapshot) if report.primary_suspect else ""
        b64_parked = image_to_base64(report.keyframe_parked)
        b64_approach = image_to_base64(report.keyframe_approach)
        b64_interaction = image_to_base64(report.keyframe_interaction)
        b64_departure = image_to_base64(report.keyframe_departure)
        b64_aftermath = image_to_base64(report.keyframe_aftermath)

        timeline_rows = "".join([
            f"""<tr>
                <td class="time-col"><code>{ev.get('time_str', 'N/A')}</code></td>
                <td><span class="badge badge-{ev.get('type', 'DEFAULT')}">{ev.get('type', '')}</span></td>
                <td>{html.escape(str(ev.get('description', '')))}</td>
            </tr>"""
            for ev in report.timeline_events
        ])

        suspect_block = ""
        if report.primary_suspect and b64_suspect:
            suspect_block = f"""
            <div class="evidence-card suspect-card">
                <div class="card-header">
                    <h3>🚨 Primary Suspect Detected (Track #{report.primary_suspect.track_id})</h3>
                    <span class="conf-tag">{report.primary_suspect.confidence_score * 100:.1f}% Confidence</span>
                </div>
                <div class="card-body grid-2">
                    <div class="crop-frame">
                        <img src="data:image/jpeg;base64,{b64_suspect}" alt="Primary Suspect" />
                        <span class="caption">Sharpest Crop (Laplacian {report.primary_suspect.best_sharpness:.1f})</span>
                    </div>
                    <div class="details">
                        <p><strong>First Contact:</strong> <code>{format_seconds(report.primary_suspect.first_contact_ts)}</code></p>
                        <p><strong>Departure Contact:</strong> <code>{format_seconds(report.primary_suspect.last_contact_ts)}</code></p>
                        <p><strong>Total Interaction:</strong> {report.primary_suspect.interaction_duration:.1f} seconds</p>
                        <p><strong>Activity:</strong> {report.primary_suspect.action_description}</p>
                        <p class="callout">This individual was physically proximate to the vehicle at the moment it departed the parking space.</p>
                    </div>
                </div>
            </div>
            """
        elif report.primary_suspect:
            suspect_block = f"""
            <div class="evidence-card suspect-card">
                <h3>🚨 Primary Suspect Track #{report.primary_suspect.track_id}</h3>
                <p>Confidence: {report.primary_suspect.confidence_score * 100:.1f}% | Activity: {report.primary_suspect.action_description}</p>
            </div>
            """
        else:
            suspect_block = "<div class='notice'>No individual was detected in immediate contact with the bike.</div>"

        bike_block = ""
        if b64_bike:
            bike_block = f"""
            <div class="crop-frame mini">
                <img src="data:image/jpeg;base64,{b64_bike}" alt="Target Bike" />
                <span class="caption">Locked Target 2-Wheeler</span>
            </div>
            """

        keyframes_html = ""
        kf_list = [
            ("1. Initial Parked State", b64_parked),
            ("2. Suspect Approach", b64_approach),
            ("3. Physical Interaction", b64_interaction),
            ("4. Bike Departure", b64_departure),
            ("5. Empty Parking Spot", b64_aftermath),
        ]
        for title, b64_img in kf_list:
            if b64_img:
                keyframes_html += f"""
                <div class="gallery-item">
                    <h4>{title}</h4>
                    <img src="data:image/jpeg;base64,{b64_img}" alt="{title}" />
                </div>
                """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCTV Forensic Dossier - Stolen Bike Investigation</title>
<style>
    :root {{
        --bg: #0f172a;
        --card: #1e293b;
        --border: #334155;
        --accent: #ef4444;
        --text: #f8fafc;
        --muted: #94a3b8;
        --success: #22c55e;
        --warning: #f59e0b;
        --blue: #38bdf8;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 24px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
    }}
    header {{
        border-bottom: 2px solid var(--border);
        padding-bottom: 20px;
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }}
    h1 {{
        margin: 0 0 6px 0;
        font-size: 26px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .status-badge {{
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 13px;
        background: #991b1b;
        color: #fef2f2;
        letter-spacing: 0.5px;
    }}
    .grid-summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .summary-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
    }}
    .summary-card h4 {{
        margin: 0 0 6px 0;
        color: var(--muted);
        font-size: 13px;
        text-transform: uppercase;
    }}
    .summary-card .val {{
        font-size: 20px;
        font-weight: 700;
        color: var(--text);
    }}
    .evidence-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
    }}
    .suspect-card {{
        border-left: 5px solid var(--accent);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }}
    .conf-tag {{
        background: #dc2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: 240px 1fr;
        gap: 20px;
    }}
    @media (max-width: 768px) {{
        .grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .crop-frame {{
        background: #020617;
        border: 2px solid #ef4444;
        border-radius: 8px;
        overflow: hidden;
        text-align: center;
    }}
    .crop-frame img {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .crop-frame .caption {{
        display: block;
        padding: 6px;
        font-size: 11px;
        color: var(--muted);
        background: #0f172a;
    }}
    .callout {{
        background: #3f1212;
        border-left: 3px solid #ef4444;
        padding: 10px;
        border-radius: 4px;
        margin-top: 14px;
        font-size: 14px;
    }}
    .gallery-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
    }}
    .gallery-item {{
        background: #0b1120;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
    }}
    .gallery-item h4 {{
        margin: 0 0 8px 0;
        font-size: 14px;
        color: var(--blue);
    }}
    .gallery-item img {{
        width: 100%;
        border-radius: 4px;
        display: block;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 14px;
    }}
    th, td {{
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        text-align: left;
    }}
    th {{
        background: #090d16;
        color: var(--muted);
        font-weight: 600;
    }}
    .badge {{
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }}
    .badge-TARGET_LOCKED {{ background: #0369a1; color: #e0f2fe; }}
    .badge-PERSON_APPROACHED {{ background: #ca8a04; color: #fefce8; }}
    .badge-DEPARTURE_DETECTED {{ background: #dc2626; color: #fef2f2; }}
    .badge-TARGET_DISAPPEARED {{ background: #b91c1c; color: #fef2f2; }}
    /* Restrained monochrome report styling; embedded CCTV frames retain their color. */
    :root {{ --bg:#fff; --card:#fff; --border:#d8d8d8; --accent:#111; --text:#111;
        --muted:#666; --success:#111; --warning:#111; --blue:#111; }}
    body {{ background:#fff !important; color:#111 !important; padding:32px 20px; }}
    .summary-card,.evidence-card,.gallery-item {{ background:#fff !important; border:1px solid #d8d8d8 !important;
        border-radius:2px !important; box-shadow:none !important; }}
    .status-badge,.conf-tag,.badge {{ background:#fff !important; color:#111 !important;
        border:1px solid #999 !important; border-radius:2px !important; }}
    .suspect-card {{ border-left:2px solid #111 !important; }}
    .crop-frame {{ background:#fff !important; border:1px solid #aaa !important; border-radius:0 !important; }}
    .crop-frame .caption {{ background:#fff !important; color:#555 !important; }}
    .gallery-item h4 {{ color:#111 !important; }}
    .gallery-item img,.crop-frame img {{ filter:none; }}
    th {{ background:#f4f4f4 !important; color:#111 !important; }}
    th,td {{ border-color:#ddd !important; }}
    .callout {{ background:#fff !important; border-left:2px solid #777 !important; }}
    @media print {{ body {{ padding:0; }} .container {{ max-width:none; }} }}
</style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>📹 CCTV Forensic Investigation Report</h1>
            <p style="color:var(--muted); margin:0;">Stolen 2-Wheeler Suspect Identification & Timeline Reconstruction</p>
        </div>
        <div>
            <span class="status-badge">{report.status}</span>
        </div>
    </header>

    <div class="grid-summary">
        <div class="summary-card">
            <h4>Target 2-Wheeler</h4>
            <div class="val">Track #{report.target_bike_id or 'N/A'}</div>
            {bike_block}
        </div>
        <div class="summary-card">
            <h4>Last Stationary</h4>
            <div class="val">{format_seconds(report.last_stationary_ts)}</div>
            <small style="color:var(--muted)">Verified resting in zone</small>
        </div>
        <div class="summary-card">
            <h4>Departure / Incident</h4>
            <div class="val" style="color:var(--accent)">{format_seconds(report.departure_ts) if report.departure_ts > 0 else 'N/A'}</div>
            <small style="color:var(--muted)">Moment vehicle moved/vanished</small>
        </div>
        <div class="summary-card">
            <h4>Footage Duration</h4>
            <div class="val">{format_seconds(report.total_duration_sec)}</div>
            <small style="color:var(--muted)">{report.total_frames_analyzed} frames analyzed</small>
        </div>
    </div>

    {suspect_block}

    <div class="evidence-card">
        <h3>🔍 Chronological Keyframe Progression</h3>
        <div class="gallery-grid">
            {keyframes_html}
        </div>
    </div>

    <div class="evidence-card">
        <h3>⏱️ Forensic Incident Timeline</h3>
        <table>
            <thead>
                <tr>
                    <th style="width: 140px;">Timecode</th>
                    <th style="width: 200px;">Event Category</th>
                    <th>Forensic Description</th>
                </tr>
            </thead>
            <tbody>
                {timeline_rows}
            </tbody>
        </table>
    </div>

    <footer style="margin-top: 40px; text-align: center; color: var(--muted); font-size: 13px;">
        Generated automatically by BikeTrace CCTV Forensics AI Toolkit &bull; Suitable for law enforcement hand-off
    </footer>
</div>
</body>
</html>
"""
        return html
