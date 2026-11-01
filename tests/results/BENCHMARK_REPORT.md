# 📊 BikeTrace AI: CCTV Video Benchmark Results

This report compiles complete empirical performance, accuracy, and forensic findings of the BikeTrace CCTV Forensics engine across **4 diverse real-world surveillance video feeds** obtained from the internet.

## 1. Executive Performance Matrix

| Case | Video Feed Source | Resolution / FPS | Total Frames | Wall Time | Speed (FPS) | Real-time Speedup | Target 2-Wheeler | Primary Suspect Identified | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1** | `test_videos/real_cctv_theft_attempt.webm` | Full CCTV | 783 frames (30.1s) | **18.12s** | **43.2 FPS** | **1.66x** | Track #3 | N/A | `STOLEN_OR_DEPARTED` |
| **Case 2** | `test_videos/intel_bicycle_surveillance.mp4` | Full CCTV | 647 frames (53.9s) | **9.43s** | **68.6 FPS** | **5.72x** | Track #3 | Track #4 (100.0%) | `STOLEN_OR_DEPARTED` |
| **Case 3** | `test_videos/real_cctv_electric_moto.webm` | Full CCTV | 428 frames (21.4s) | **34.04s** | **12.6 FPS** | **0.63x** | Track #3 | Track #2 (100.0%) | `STOLEN_OR_DEPARTED` |
| **Case 4** | `sample_test_cctv.mp4` | Full CCTV | 350 frames (14.0s) | **6.23s** | **56.2 FPS** | **2.25x** | Track #1 | Track #4 (100.0%) | `STOLEN_OR_DEPARTED` |

---

## 2. Detailed Case-by-Case Findings

### Case 1: Residential Compound Motorcycle Theft Attempt
- **Video Source:** `test_videos/real_cctv_theft_attempt.webm`
- **Frames / Duration:** 783 frames (00:00:30.115)
- **Processing Wall Time:** `18.12 seconds` (**43.2 FPS**, **1.66x faster than real-time**)
- **Target 2-Wheeler Locked:** `Track #3`
- **Incident Status:** `STOLEN_OR_DEPARTED`
- **Last Verified Stationary:** `00:00:23.615`
- **Departure / Action Timestamp:** `00:00:23.692`
- **Primary Suspect Identified:** Track #None (Confidence: `0.0%`)
- **Observed Suspect Action:** None
- **Forensic Dossier Location:** [`test_results/case1_theft_attempt/incident_report.html`](file:///D:\github_d\bikestolen\test_results\case1_theft_attempt/incident_report.html)
- **Annotated Evidence Video:** [`test_results/case1_theft_attempt/forensics_annotated_scan.mp4`](file:///D:\github_d\bikestolen\test_results\case1_theft_attempt/forensics_annotated_scan.mp4)
- **Snapshots Extracted:** 3 evidence images in `test_results/case1_theft_attempt/snapshots/`

### Case 2: Intel IoT Surveillance Street & Bicycle Feed
- **Video Source:** `test_videos/intel_bicycle_surveillance.mp4`
- **Frames / Duration:** 647 frames (00:00:53.917)
- **Processing Wall Time:** `9.43 seconds` (**68.6 FPS**, **5.72x faster than real-time**)
- **Target 2-Wheeler Locked:** `Track #3`
- **Incident Status:** `STOLEN_OR_DEPARTED`
- **Last Verified Stationary:** `00:00:27.500`
- **Departure / Action Timestamp:** `00:00:27.667`
- **Primary Suspect Identified:** Track #4 (Confidence: `100.0%`)
- **Observed Suspect Action:** Interacted with bike at departure
- **Forensic Dossier Location:** [`test_results/case2_intel_bicycle/incident_report.html`](file:///D:\github_d\bikestolen\test_results\case2_intel_bicycle/incident_report.html)
- **Annotated Evidence Video:** [`test_results/case2_intel_bicycle/forensics_annotated_scan.mp4`](file:///D:\github_d\bikestolen\test_results\case2_intel_bicycle/forensics_annotated_scan.mp4)
- **Snapshots Extracted:** 7 evidence images in `test_results/case2_intel_bicycle/snapshots/`

### Case 3: Overhead Security Camera 2-Wheeler Intercept
- **Video Source:** `test_videos/real_cctv_electric_moto.webm`
- **Frames / Duration:** 428 frames (00:00:21.400)
- **Processing Wall Time:** `34.04 seconds` (**12.6 FPS**, **0.63x faster than real-time**)
- **Target 2-Wheeler Locked:** `Track #3`
- **Incident Status:** `STOLEN_OR_DEPARTED`
- **Last Verified Stationary:** `00:00:12.800`
- **Departure / Action Timestamp:** `00:00:12.850`
- **Primary Suspect Identified:** Track #2 (Confidence: `100.0%`)
- **Observed Suspect Action:** Interacted with bike at departure
- **Forensic Dossier Location:** [`test_results/case3_electric_moto/incident_report.html`](file:///D:\github_d\bikestolen\test_results\case3_electric_moto/incident_report.html)
- **Annotated Evidence Video:** [`test_results/case3_electric_moto/forensics_annotated_scan.mp4`](file:///D:\github_d\bikestolen\test_results\case3_electric_moto/forensics_annotated_scan.mp4)
- **Snapshots Extracted:** 6 evidence images in `test_results/case3_electric_moto/snapshots/`

### Case 4: End-to-End Parking Lot CCTV Theft Benchmark
- **Video Source:** `sample_test_cctv.mp4`
- **Frames / Duration:** 350 frames (00:00:14.000)
- **Processing Wall Time:** `6.23 seconds` (**56.2 FPS**, **2.25x faster than real-time**)
- **Target 2-Wheeler Locked:** `Track #1`
- **Incident Status:** `STOLEN_OR_DEPARTED`
- **Last Verified Stationary:** `00:00:09.760`
- **Departure / Action Timestamp:** `00:00:09.840`
- **Primary Suspect Identified:** Track #4 (Confidence: `100.0%`)
- **Observed Suspect Action:** Interacted with bike at departure
- **Forensic Dossier Location:** [`test_results/case4_parking_theft/incident_report.html`](file:///D:\github_d\bikestolen\test_results\case4_parking_theft/incident_report.html)
- **Annotated Evidence Video:** [`test_results/case4_parking_theft/forensics_annotated_scan.mp4`](file:///D:\github_d\bikestolen\test_results\case4_parking_theft/forensics_annotated_scan.mp4)
- **Snapshots Extracted:** 7 evidence images in `test_results/case4_parking_theft/snapshots/`
