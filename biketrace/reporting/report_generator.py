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
            "# ðŸš¨ CCTV FORENSIC INCIDENT REPORT: STOLEN 2-WHEELER INVESTIGATION",
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
        """Render a concise, responsive, self-contained monochrome evidence report."""
        esc = lambda value: html.escape(str(value))
        image = lambda value, alt: (
            f'<img src="data:image/jpeg;base64,{value}" alt="{esc(alt)}">' if value else ""
        )

        timeline_rows = "".join(
            f"""<tr>
                <td class="time">{esc(event.get('time_str', 'N/A'))}</td>
                <td><span class="event-tag">{esc(event.get('type', 'Event'))}</span></td>
                <td>{esc(event.get('description', ''))}</td>
            </tr>"""
            for event in report.timeline_events
        )
        if not timeline_rows:
            timeline_rows = '<tr><td colspan="3" class="empty">No timeline events were recorded.</td></tr>'

        keyframes = [
            ("Parked", report.keyframe_parked),
            ("Approach", report.keyframe_approach),
            ("Interaction", report.keyframe_interaction),
            ("Departure", report.keyframe_departure),
            ("Aftermath", report.keyframe_aftermath),
        ]
        gallery = "".join(
            f'<figure class="frame">{image(image_to_base64(frame), title)}<figcaption>{esc(title)}</figcaption></figure>'
            for title, frame in keyframes if frame is not None and frame.size
        )
        if not gallery:
            gallery = '<p class="empty">No keyframes were captured.</p>'

        bike_image = image_to_base64(report.target_initial_crop)
        target_evidence = (
            f'<figure class="portrait">{image(bike_image, "Target vehicle")}'
            '<figcaption>Target vehicle</figcaption></figure>' if bike_image else ""
        )
        suspect = report.primary_suspect
        suspect_image = image_to_base64(suspect.best_snapshot) if suspect else ""
        if suspect:
            suspect_details = f"""
                <div class="profile">
                    <div class="profile-heading"><h3>Track {esc(suspect.track_id)}</h3>
                    <span class="confidence">{suspect.confidence_score * 100:.1f}% confidence</span></div>
                    <dl>
                        <div><dt>First contact</dt><dd>{esc(format_seconds(suspect.first_contact_ts))}</dd></div>
                        <div><dt>Last contact</dt><dd>{esc(format_seconds(suspect.last_contact_ts))}</dd></div>
                        <div><dt>Interaction</dt><dd>{suspect.interaction_duration:.1f} seconds</dd></div>
                        <div><dt>Observed activity</dt><dd>{esc(suspect.action_description)}</dd></div>
                    </dl>
                </div>"""
            suspect_evidence = (
                f'<figure class="portrait">{image(suspect_image, "Suspect")}'
                '<figcaption>Best available frame</figcaption></figure>' if suspect_image else ""
            )
        else:
            suspect_details = '<p class="empty">No suspect was identified in the analyzed footage.</p>'
            suspect_evidence = ""

        departure = format_seconds(report.departure_ts) if report.departure_ts > 0 else "Not detected"
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Incident report | BikeTrace</title>
<style>
    :root {{ color-scheme: light; --ink:#171717; --muted:#686868; --line:#dedede; --soft:#f6f6f5; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#fff; color:var(--ink); font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(100% - 40px, 1040px); margin:0 auto; padding:48px 0 64px; }}
    .masthead {{ display:flex; justify-content:space-between; gap:24px; align-items:center; padding-bottom:16px; border-bottom:1px solid var(--line); }}
    .wordmark {{ font-size:12px; letter-spacing:.13em; font-weight:650; text-transform:uppercase; }}
    .document-type {{ color:var(--muted); font-size:13px; }}
    .intro {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px; padding:32px 0 28px; }}
    h1 {{ margin:0; font-size:clamp(30px,5vw,42px); line-height:1.12; letter-spacing:-.045em; font-weight:620; }}
    .subhead {{ margin:9px 0 0; color:var(--muted); }}
    .status {{ flex:none; border:1px solid var(--line); border-radius:999px; padding:6px 11px; font-size:11px; letter-spacing:.08em; font-weight:650; text-transform:uppercase; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:38px; }}
    .metric {{ min-height:91px; padding:16px 18px; border-right:1px solid var(--line); }}
    .metric:last-child {{ border:0; }}
    .label,dt {{ display:block; color:var(--muted); font-size:11px; letter-spacing:.07em; text-transform:uppercase; }}
    .value {{ display:block; margin-top:7px; font-size:18px; font-weight:590; letter-spacing:-.02em; }}
    section {{ margin-top:36px; }}
    .section-heading {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0; font-size:17px; font-weight:600; letter-spacing:-.02em; }}
    .section-note {{ color:var(--muted); font-size:12px; }}
    .evidence-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }}
    .evidence-panel {{ min-width:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .evidence-content {{ display:flex; gap:18px; padding:16px; align-items:flex-start; }}
    figure {{ margin:0; }}
    .portrait {{ flex:0 0 38%; min-width:120px; }}
    .portrait img {{ display:block; width:100%; max-height:270px; object-fit:contain; background:var(--soft); }}
    figcaption {{ padding-top:7px; color:var(--muted); font-size:12px; }}
    .profile {{ flex:1; min-width:0; }}
    .profile-heading {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; }}
    h3 {{ margin:0; font-size:15px; font-weight:600; }}
    .confidence {{ color:var(--muted); font-size:12px; }}
    dl {{ margin:12px 0 0; }}
    dl div {{ padding:8px 0; border-top:1px solid var(--line); }}
    dd {{ margin:2px 0 0; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
    .frame {{ min-width:0; padding:8px; border:1px solid var(--line); border-radius:7px; }}
    .frame img {{ display:block; width:100%; aspect-ratio:16/10; object-fit:contain; background:var(--soft); }}
    .frame figcaption {{ color:var(--ink); font-weight:550; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; text-align:left; }}
    th {{ padding:10px 14px; background:var(--soft); color:var(--muted); font-size:11px; font-weight:600; letter-spacing:.07em; text-transform:uppercase; }}
    td {{ padding:12px 14px; border-top:1px solid var(--line); vertical-align:top; }}
    .time {{ width:145px; white-space:nowrap; font-variant-numeric:tabular-nums; }}
    .event-tag {{ display:inline-block; padding:2px 7px; border:1px solid var(--line); border-radius:4px; font-size:11px; }}
    .empty {{ padding:16px; color:var(--muted); }}
    footer {{ margin-top:42px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }}
    @media(max-width:700px) {{ main {{ width:min(100% - 28px,1040px); padding-top:24px; }} .intro {{ align-items:flex-start; flex-direction:column; }} .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .metric:nth-child(2) {{ border-right:0; }} .metric:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} .evidence-grid {{ grid-template-columns:1fr; }} }}
    @media(max-width:430px) {{ .evidence-content {{ flex-direction:column; }} .portrait {{ width:100%; max-width:260px; flex-basis:auto; }} .portrait img {{ max-height:340px; }} }}
    @media print {{ main {{ width:100%; padding:0; }} .evidence-panel,.frame,.metrics {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<main>
    <div class="masthead"><span class="wordmark">BikeTrace</span><span class="document-type">CCTV incident analysis</span></div>
    <header class="intro"><div><h1>Incident report</h1><p class="subhead">Vehicle departure and associated activity</p></div><span class="status">{esc(report.status)}</span></header>
    <div class="metrics" aria-label="Incident summary">
        <div class="metric"><span class="label">Target vehicle</span><span class="value">{f'Track {esc(report.target_bike_id)}' if report.target_bike_id else 'Not identified'}</span></div>
        <div class="metric"><span class="label">Last stationary</span><span class="value">{esc(format_seconds(report.last_stationary_ts))}</span></div>
        <div class="metric"><span class="label">Departure</span><span class="value">{esc(departure)}</span></div>
        <div class="metric"><span class="label">Footage analyzed</span><span class="value">{esc(format_seconds(report.total_duration_sec))}</span></div>
    </div>
    <section aria-labelledby="evidence-title">
        <div class="section-heading"><h2 id="evidence-title">People and vehicle</h2><span class="section-note">Color frames preserved from source footage</span></div>
        <div class="evidence-grid">
            <article class="evidence-panel"><div class="evidence-content">{target_evidence}<div class="profile"><h3>Target vehicle</h3><dl><div><dt>Track</dt><dd>{esc(report.target_bike_id or 'Not identified')}</dd></div><div><dt>Last stationary</dt><dd>{esc(format_seconds(report.last_stationary_ts))}</dd></div><div><dt>Departure</dt><dd>{esc(departure)}</dd></div></dl></div></div></article>
            <article class="evidence-panel"><div class="evidence-content">{suspect_evidence}{suspect_details}</div></article>
        </div>
    </section>
    <section aria-labelledby="frames-title"><div class="section-heading"><h2 id="frames-title">Keyframes</h2><span class="section-note">{report.total_frames_analyzed} frames analyzed</span></div><div class="gallery">{gallery}</div></section>
    <section aria-labelledby="timeline-title"><div class="section-heading"><h2 id="timeline-title">Timeline</h2></div><div class="table-wrap"><table><thead><tr><th>Time</th><th>Event</th><th>Description</th></tr></thead><tbody>{timeline_rows}</tbody></table></div></section>
    <footer>Generated by BikeTrace CCTV Forensics</footer>
</main>
</body>
</html>"""
