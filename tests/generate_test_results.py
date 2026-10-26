"""
Automated Visual Test Results Generator (tests/generate_test_results.py).
Executes end-to-end detection on benchmark media and generates annotated verification
artifacts in tests/test_results/ for Human-in-the-Loop (HITL) Gate 5 review.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import DriverDrowsinessDetector
from src.utils import draw_drowsiness_overlay


def run_visual_verification():
    out_dir = PROJECT_ROOT / "tests" / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    presets_dir = PROJECT_ROOT / "data" / "sample_presets"
    detector = DriverDrowsinessDetector(static_image_mode=True)

    test_targets = [
        ("driver_alert_1.jpg", "verified_alert_overlay.jpg"),
        ("driver_yawn_1.jpg", "verified_yawn_overlay.jpg"),
        ("driver_microsleep_1.jpg", "verified_microsleep_overlay.jpg")
    ]

    print("==================================================================")
    print("RUNNING AUTOMATED VISUAL QA VERIFICATION BATCH")
    print("==================================================================")

    for src_name, out_name in test_targets:
        img_path = presets_dir / src_name
        if not img_path.exists():
            print(f"[SKIP] Preset not found: {src_name}")
            continue

        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detector.reset()
        res = detector.process_frame(rgb, consec_frames=1)

        annotated_rgb = draw_drowsiness_overlay(rgb, res["landmarks_px"], res)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        save_path = out_dir / out_name
        cv2.imwrite(str(save_path), annotated_bgr)

        print(f"[OK] Generated: {out_name}")
        print(f"   Face: {res['face_detected']} | Status: {res['driver_status']}")
        print(f"   EAR: {res['avg_ear']} | MAR: {res['mar']} | Pitch: {res['head_pitch_deg']} deg | Latency: {res['inference_latency_ms']:.1f}ms")
        print()

    detector.close()
    print(f"All visual verification artifacts generated in: {out_dir}")


if __name__ == "__main__":
    run_visual_verification()
