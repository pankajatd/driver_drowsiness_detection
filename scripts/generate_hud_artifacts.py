import os
import sys
import cv2

sys.path.append(r'C:\Pankaja\AUTOMATED_SDLC_AGENT_PROJECTS\driver_drowsiness_detection')
from src.detector import DriverDrowsinessDetector
from src.utils import draw_drowsiness_overlay

preset_dir = r'C:\Pankaja\AUTOMATED_SDLC_AGENT_PROJECTS\driver_drowsiness_detection\data\sample_presets'
art_dir = r'C:\Users\panka\.gemini\antigravity\brain\ce6e92d6-586e-4366-aa56-1c9b40d5054a'

pairs = [
    ('driver_alert_1.jpg', 'hud_preset_alert.jpg'),
    ('driver_yawn_1.jpg', 'hud_preset_yawn.jpg'),
    ('driver_microsleep_1.jpg', 'hud_preset_microsleep.jpg')
]

for name, prefix in pairs:
    raw_bgr = cv2.imread(os.path.join(preset_dir, name))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    det = DriverDrowsinessDetector(static_image_mode=True)
    res = det.process_frame(raw_rgb, consec_frames=1)
    det.close()
    annotated_rgb = draw_drowsiness_overlay(raw_rgb, res['landmarks_px'], res)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_BGR2RGB)
    out_path = os.path.join(art_dir, prefix)
    cv2.imwrite(out_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    st = res.get('driver_status')
    print("Generated " + prefix + " (status=" + str(st) + ")")
