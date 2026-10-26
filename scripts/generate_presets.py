"""
Sample Preset Video & Image Generator for Driver Drowsiness Detection
Generates benchmark test media in data/sample_presets/:
1. driver_alert.mp4 & driver_alert_1.jpg
2. driver_yawn.mp4 & driver_yawn_1.jpg
3. driver_microsleep.mp4 & driver_microsleep_1.jpg
"""

import os
import cv2
import numpy as np

def draw_driver_face(
    canvas_w=640,
    canvas_h=480,
    eye_state="OPEN",     # "OPEN", "BLINK", "CLOSED"
    mouth_state="CLOSED", # "CLOSED", "TALKING", "YAWN"
    head_pitch=0.0        # degrees downward
):
    # Dark automotive cabin background
    img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)
    img[:, :] = (30, 35, 45) # Dark slate cabin interior

    # Car headrest in background
    cv2.rectangle(img, (200, 40), (440, 220), (50, 55, 65), -1)
    # Seatbelt
    cv2.line(img, (120, 0), (520, 480), (20, 20, 25), 24)

    # Center of head with pitch offset
    cx = canvas_w // 2
    cy = int(canvas_h // 2 + head_pitch * 2.5)

    # Face Oval (Skin tone)
    skin_color = (215, 185, 160)
    cv2.ellipse(img, (cx, cy), (130, 175), 0, 0, 360, skin_color, -1)

    # Hair
    cv2.ellipse(img, (cx, cy - 80), (135, 100), 0, 180, 360, (40, 30, 25), -1)

    # Eyebrows
    cv2.line(img, (cx - 80, cy - 45), (cx - 25, cy - 40), (40, 30, 25), 5)
    cv2.line(img, (cx + 25, cy - 40), (cx + 80, cy - 45), (40, 30, 25), 5)

    # Nose
    cv2.line(img, (cx, cy - 20), (cx - 8, cy + 25), (180, 140, 120), 3)
    cv2.line(img, (cx - 8, cy + 25), (cx + 8, cy + 25), (180, 140, 120), 3)

    # EYES
    eye_y = cy - 20
    left_x = cx - 55
    right_x = cx + 55

    if eye_state == "OPEN":
        # Left eye open
        cv2.ellipse(img, (left_x, eye_y), (28, 14), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (left_x, eye_y), 9, (45, 35, 30), -1)
        cv2.circle(img, (left_x, eye_y), 4, (10, 10, 10), -1)
        cv2.circle(img, (left_x - 3, eye_y - 3), 2, (255, 255, 255), -1)
        # Right eye open
        cv2.ellipse(img, (right_x, eye_y), (28, 14), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (right_x, eye_y), 9, (45, 35, 30), -1)
        cv2.circle(img, (right_x, eye_y), 4, (10, 10, 10), -1)
        cv2.circle(img, (right_x - 3, eye_y - 3), 2, (255, 255, 255), -1)
    elif eye_state == "BLINK":
        # Half closed blink
        cv2.ellipse(img, (left_x, eye_y), (28, 5), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (left_x, eye_y), 5, (45, 35, 30), -1)
        cv2.ellipse(img, (right_x, eye_y), (28, 5), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (right_x, eye_y), 5, (45, 35, 30), -1)
    else: # "CLOSED"
        # Eyelids completely closed
        cv2.line(img, (left_x - 26, eye_y), (left_x + 26, eye_y), (60, 45, 40), 4)
        cv2.line(img, (right_x - 26, eye_y), (right_x + 26, eye_y), (60, 45, 40), 4)

    # MOUTH
    mouth_y = cy + 70
    if mouth_state == "CLOSED":
        cv2.line(img, (cx - 40, mouth_y), (cx + 40, mouth_y), (140, 65, 65), 5)
    elif mouth_state == "TALKING":
        cv2.ellipse(img, (cx, mouth_y), (35, 12), 0, 0, 360, (140, 65, 65), -1)
    else: # "YAWN"
        # Wide open yawning mouth
        cv2.ellipse(img, (cx, mouth_y), (42, 38), 0, 0, 360, (50, 20, 20), -1)
        cv2.ellipse(img, (cx, mouth_y), (46, 42), 0, 0, 360, (140, 65, 65), 4)

    return img

def generate_video(filename, sequence_desc, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    for eye_s, mouth_s, pitch, count in sequence_desc:
        for _ in range(count):
            frame = draw_driver_face(eye_state=eye_s, mouth_state=mouth_s, head_pitch=pitch)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Generated video: {filename}")

def run():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_presets"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Alert Driver
    alert_img = draw_driver_face(eye_state="OPEN", mouth_state="CLOSED")
    cv2.imwrite(os.path.join(out_dir, "driver_alert_1.jpg"), cv2.cvtColor(alert_img, cv2.COLOR_RGB2BGR))
    # Video: Open eyes, occasional fast blinks (3 frames blink every 40 frames)
    seq_alert = [
        ("OPEN", "CLOSED", 0.0, 35),
        ("BLINK", "CLOSED", 0.0, 3),
        ("OPEN", "CLOSED", 0.0, 45),
        ("BLINK", "CLOSED", 0.0, 4),
        ("OPEN", "CLOSED", 0.0, 35)
    ]
    generate_video(os.path.join(out_dir, "driver_alert.mp4"), seq_alert)

    # 2. Yawning Driver
    yawn_img = draw_driver_face(eye_state="OPEN", mouth_state="YAWN")
    cv2.imwrite(os.path.join(out_dir, "driver_yawn_1.jpg"), cv2.cvtColor(yawn_img, cv2.COLOR_RGB2BGR))
    seq_yawn = [
        ("OPEN", "CLOSED", 0.0, 25),
        ("OPEN", "YAWN", 0.0, 45), # Sustained yawn
        ("OPEN", "CLOSED", 0.0, 30)
    ]
    generate_video(os.path.join(out_dir, "driver_yawn.mp4"), seq_yawn)

    # 3. Microsleep Driver
    sleep_img = draw_driver_face(eye_state="CLOSED", mouth_state="CLOSED", head_pitch=12.0)
    cv2.imwrite(os.path.join(out_dir, "driver_microsleep_1.jpg"), cv2.cvtColor(sleep_img, cv2.COLOR_RGB2BGR))
    # Video: Eyes open for 15 frames, then close continuously for 65 frames (> 2.0s) with head slumping down
    seq_sleep = [
        ("OPEN", "CLOSED", 0.0, 15),
        ("CLOSED", "CLOSED", 10.0, 65) # Closed eyes + head slump = Critical alarm!
    ]
    generate_video(os.path.join(out_dir, "driver_microsleep.mp4"), seq_sleep)

    print("All sample presets generated successfully in:", out_dir)

if __name__ == "__main__":
    run()
