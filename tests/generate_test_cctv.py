"""Generate a realistic synthetic CCTV test scenario with parked motorcycle, bystander, and bike theft event."""
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def create_realistic_test_cctv(output_video_path: str = "tests/videos/sample_test_cctv.mp4", duration_sec: int = 14, fps: int = 25):
    """
    Synthesize an authentic-looking CCTV video:
    - CCTV camera timestamp and camera ID overlay (e.g. CAM-04 REAR PARKING)
    - Parked motorcycle in designated bay
    - Innocent pedestrian passing in background (non-interacting)
    - Suspect approaching, interacting with the motorcycle, and riding off
    - Empty parking bay aftermath
    """
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 540
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Base background: Asphalt parking lot with pavement and bay lines
    bg = np.ones((height, width, 3), dtype=np.uint8) * 110
    # Pavement sidewalk at top
    cv2.rectangle(bg, (0, 0), (width, 140), (160, 160, 160), -1)
    cv2.line(bg, (0, 140), (width, 140), (70, 70, 70), 3)

    # Parking bay painted lines
    for bx in [150, 380, 610, 840]:
        cv2.line(bg, (bx, 140), (bx, 480), (220, 220, 220), 4)
    # Parking bay numbers
    cv2.putText(bg, "BAY 1", (230, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(bg, "BAY 2", (460, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(bg, "BAY 3", (690, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Load real motorcycle and person textures
    assets_dir = Path(__file__).parent / "assets"
    moto_img = cv2.imread(str(assets_dir / "test_motorcycle.jpg"))
    if moto_img is not None:
        moto_resized = cv2.resize(moto_img, (180, 130))
    else:
        # Fallback drawn motorcycle
        moto_resized = np.zeros((130, 180, 3), dtype=np.uint8)
        cv2.circle(moto_resized, (35, 95), 30, (20, 20, 20), -1)
        cv2.circle(moto_resized, (145, 95), 30, (20, 20, 20), -1)
        cv2.rectangle(moto_resized, (35, 60), (145, 90), (0, 0, 200), -1)

    person_src = cv2.imread(str(assets_dir / "test_zidane.jpg"))
    if person_src is not None:
        # Crop person from zidane
        person_crop = person_src[100:600, 150:450]
        suspect_img = cv2.resize(person_crop, (75, 170))
        bystander_img = cv2.resize(person_crop, (50, 110))
    else:
        suspect_img = np.zeros((170, 75, 3), dtype=np.uint8)
        cv2.circle(suspect_img, (37, 25), 20, (200, 180, 150), -1)
        cv2.rectangle(suspect_img, (15, 45), (60, 130), (50, 50, 180), -1)
        bystander_img = cv2.resize(suspect_img, (50, 110))

    # Initial parking coordinates for the target bike in Bay 2
    bike_init_x = 440
    bike_init_y = 260
    mw, mh = moto_resized.shape[1], moto_resized.shape[0]
    sw, sh = suspect_img.shape[1], suspect_img.shape[0]
    bw, bh = bystander_img.shape[1], bystander_img.shape[0]

    base_time = datetime(2026, 10, 31, 14, 15, 0)

    for f_idx in range(total_frames):
        curr_time = base_time + timedelta(seconds=(f_idx / fps))
        frame = bg.copy()

        # Phases:
        # 1. 0.0s - 2.5s (f: 0-62): Bike parked, peaceful scene
        # 2. 2.5s - 5.0s (f: 63-125): Innocent bystander walks along pavement (y=20, x from 50 to 800)
        # 3. 5.0s - 7.5s (f: 126-187): Suspect enters from right (x: 880 -> 400, y: 240)
        # 4. 7.5s - 9.5s (f: 188-237): Suspect stands over bike tampering
        # 5. 9.5s - 12.0s (f: 238-300): Suspect and bike move together to left (x: 440 -> -200)
        # 6. 12.0s - 14.0s (f: 301-350): Empty bay aftermath

        # Bystander animation
        if 62 <= f_idx <= 130:
            b_progress = (f_idx - 62) / (130 - 62)
            bx = int(50 + b_progress * 750)
            by = 25
            frame[by:by+bh, bx:bx+bw] = bystander_img

        # Suspect & Bike animation
        if f_idx < 238:
            # Bike is parked in Bay 2
            frame[bike_init_y:bike_init_y+mh, bike_init_x:bike_init_x+mw] = moto_resized

            if 125 <= f_idx < 188:
                # Suspect walking toward bike
                s_prog = (f_idx - 125) / (188 - 125)
                sx = int(880 - s_prog * 470)  # lands at 410 (right next to bike)
                sy = 240
                frame[sy:sy+sh, sx:sx+sw] = suspect_img
            elif 188 <= f_idx < 238:
                # Suspect actively tampering right next to bike
                sx = 410
                sy = 240
                frame[sy:sy+sh, sx:sx+sw] = suspect_img
        elif 238 <= f_idx <= 300:
            # Suspect riding off with bike to the left
            dep_prog = (f_idx - 238) / (300 - 238)
            curr_x = int(bike_init_x - dep_prog * 680)
            curr_y = int(bike_init_y + dep_prog * 30)

            # Draw bike and rider leaving
            if curr_x + mw > 0 and curr_x < width:
                draw_x1 = max(0, curr_x)
                draw_x2 = min(width, curr_x + mw)
                crop_x1 = draw_x1 - curr_x
                crop_x2 = crop_x1 + (draw_x2 - draw_x1)

                frame[curr_y:curr_y+mh, draw_x1:draw_x2] = moto_resized[:, crop_x1:crop_x2]

                # Suspect riding on top/behind bike
                rsx = curr_x + 10
                rsy = curr_y - 40
                if rsx + sw > 0 and rsx < width and rsy + sh > 0 and rsy < height:
                    s_x1 = max(0, rsx)
                    s_x2 = min(width, rsx + sw)
                    s_y1 = max(0, rsy)
                    s_y2 = min(height, rsy + sh)
                    frame[s_y1:s_y2, s_x1:s_x2] = suspect_img[:(s_y2-s_y1), :(s_x2-s_x1)]
        else:
            # Bike has left! Empty bay aftermath
            pass

        # Add realistic CCTV Noise and Timestamp Overlay
        # Timestamp bar at top
        time_str = curr_time.strftime("%Y-%m-%d  %H:%M:%S")
        hud_bar = f"CAM-02 PARKING EAST | {time_str} | REC"
        cv2.rectangle(frame, (10, 10), (520, 36), (0, 0, 0), -1)
        cv2.putText(frame, hud_bar, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # Red REC indicator dot
        cv2.circle(frame, (505, 23), 5, (0, 0, 255), -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic CCTV video: {output_video_path} ({total_frames} frames, {duration_sec}s @ {fps}fps)")
    return output_video_path


if __name__ == "__main__":
    create_realistic_test_cctv("tests/videos/sample_test_cctv.mp4")
