import cv2
import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.detector import DriverDrowsinessDetector

preset_dir = os.path.join(root_dir, "data", "sample_presets")

def standardize_image(img, target_w=640, target_h=480):
    """Crops and resizes image to standard automotive 640x480 resolution."""
    h, w = img.shape[:2]
    # Crop to 4:3 aspect ratio around center
    target_aspect = target_w / target_h
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Too wide, crop width
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        img = img[:, start_x:start_x + new_w]
    else:
        # Too tall, crop height favoring top (face is usually upper center)
        new_h = int(w / target_aspect)
        start_y = max(0, (h - new_h) // 3)
        img = img[start_y:start_y + new_h, :]

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

# 1. Standardize and rewrite the 3 real images
alert_raw = cv2.imread(os.path.join(preset_dir, "driver_alert_1.jpg"))
yawn_raw = cv2.imread(os.path.join(preset_dir, "driver_yawn_1.jpg"))
sleep_raw = cv2.imread(os.path.join(preset_dir, "driver_microsleep_1.jpg"))

alert_img = standardize_image(alert_raw)
yawn_img = standardize_image(yawn_raw)
sleep_img = standardize_image(sleep_raw)

cv2.imwrite(os.path.join(preset_dir, "driver_alert_1.jpg"), alert_img)
cv2.imwrite(os.path.join(preset_dir, "driver_yawn_1.jpg"), yawn_img)
cv2.imwrite(os.path.join(preset_dir, "driver_microsleep_1.jpg"), sleep_img)

print("Standardized real images to 640x480.")

# Helper to apply subtle road vibration / camera motion
def add_subtle_motion(img, dx=0, dy=0, brightness_factor=1.0):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if brightness_factor != 1.0:
        shifted = np.clip(shifted.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    return shifted

# Helper to write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 1. Generate driver_alert.mp4 (Real alert driver, 90 frames = 3.0s at 30 fps)
# Features natural subtle road vibration and two natural micro-blinks (2 frames each)
video_alert_path = os.path.join(preset_dir, "driver_alert.mp4")
out_alert = cv2.VideoWriter(video_alert_path, fourcc, 30.0, (640, 480))

for i in range(90):
    dx = int(np.sin(i * 0.4) * 1.5)
    dy = int(np.cos(i * 0.3) * 1.0)
    lum = 1.0 + 0.02 * np.sin(i * 0.1)
    
    # At frame 25-26 and frame 65-66, simulate brief 2-frame blink by blending with closed eyes
    if 25 <= i <= 26 or 65 <= i <= 66:
        frame = cv2.addWeighted(alert_img, 0.2, sleep_img, 0.8, 0)
    else:
        frame = alert_img

    frame = add_subtle_motion(frame, dx, dy, lum)
    out_alert.write(frame)

out_alert.release()
print(f"Generated {video_alert_path}")

# 2. Generate driver_yawn.mp4 (Real driver yawn, 90 frames = 3.0s at 30 fps)
# 0-25: alert, 25-35: transition to yawn, 35-70: sustained real yawn, 70-90: recovery
video_yawn_path = os.path.join(preset_dir, "driver_yawn.mp4")
out_yawn = cv2.VideoWriter(video_yawn_path, fourcc, 30.0, (640, 480))

for i in range(90):
    dx = int(np.sin(i * 0.3) * 1.5)
    dy = int(np.cos(i * 0.2) * 1.0)
    
    if i < 25:
        frame = alert_img
    elif 25 <= i < 35:
        alpha = (i - 25) / 10.0
        frame = cv2.addWeighted(alert_img, 1.0 - alpha, yawn_img, alpha, 0)
    elif 35 <= i < 70:
        frame = yawn_img
    elif 70 <= i < 80:
        alpha = (80 - i) / 10.0
        frame = cv2.addWeighted(alert_img, 1.0 - alpha, yawn_img, alpha, 0)
    else:
        frame = alert_img

    frame = add_subtle_motion(frame, dx, dy)
    out_yawn.write(frame)

out_yawn.release()
print(f"Generated {video_yawn_path}")

# 3. Generate driver_microsleep.mp4 (Real driver microsleep, 90 frames = 3.0s at 30 fps)
# 0-15: alert, 15-25: eyes close & head slumps down, 25-90: sustained closed eyes & slump (>2s) -> Critical Siren!
video_sleep_path = os.path.join(preset_dir, "driver_microsleep.mp4")
out_sleep = cv2.VideoWriter(video_sleep_path, fourcc, 30.0, (640, 480))

for i in range(90):
    if i < 15:
        dx = int(np.sin(i * 0.4) * 1.2)
        dy = int(np.cos(i * 0.3) * 1.0)
        frame = add_subtle_motion(alert_img, dx, dy)
    elif 15 <= i < 28:
        alpha = (i - 15) / 13.0
        blended = cv2.addWeighted(alert_img, 1.0 - alpha, sleep_img, alpha, 0)
        # Head nodding slump downwards
        slump_dy = int(alpha * 12)
        frame = add_subtle_motion(blended, 0, slump_dy)
    else:
        # Full microsleep: closed eyes and slumped head
        slump_dy = 12 + int(np.sin(i * 0.1) * 2)
        frame = add_subtle_motion(sleep_img, 0, slump_dy)

    out_sleep.write(frame)

out_sleep.release()
print(f"Generated {video_sleep_path}")
