"""
Interactive Web Dashboard & Driver Console Presentation Layer
Project: Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System
System Codename: DrowsyGuard DMS & Automotive Telemetry Engine
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root and src directory are in sys.path regardless of execution directory
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import streamlit as st
import numpy as np
import cv2

# Import local subsystems
try:
    from src.detector import DriverDrowsinessDetector
    from src.utils import (
        validate_media_input,
        load_and_validate_image,
        extract_video_frame_iterator,
        draw_drowsiness_overlay,
        generate_audio_alert_b64,
        format_telemetry_csv
    )
except ImportError:
    from detector import DriverDrowsinessDetector
    from utils import (
        validate_media_input,
        load_and_validate_image,
        extract_video_frame_iterator,
        draw_drowsiness_overlay,
        generate_audio_alert_b64,
        format_telemetry_csv
    )

# =============================================================================
# Streamlit Application Configuration
# =============================================================================
st.set_page_config(
    page_title="DrowsyGuard | Real-Time Driver Drowsiness & Fatigue Detection",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Automotive Dark-Slate CSS Theme
st.markdown("""
<style>
    .main {
        background-color: #0b1120;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.50rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-attentive {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-drowsy {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-critical {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
        animation: pulse 1.2s infinite;
    }
    .badge-searching {
        background-color: #374151;
        color: #9ca3af;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper: Render Plain-English Metric Cards
# =============================================================================
def render_metric_cards(result: Dict[str, Any], col1, col2, col3, col4):
    """Renders human-friendly driver monitoring telemetry cards."""
    status = result.get("driver_status", "ATTENTIVE")
    avg_ear = float(result.get("avg_ear", 0.0))
    mar = float(result.get("mar", 0.0))
    pitch = float(result.get("head_pitch_deg", 0.0))
    is_ms = bool(result.get("microsleep_alert", False))
    is_yawn = bool(result.get("yawn_alert", False))
    face_detected = bool(result.get("face_detected", False))

    # 1. Driver Alertness Status
    if is_ms or status == "CRITICAL":
        status_html = '<span class="badge-critical">🚨 CRITICAL (Microsleep Alarm!)</span>'
    elif is_yawn or status == "DROWSY":
        status_html = '<span class="badge-drowsy">🟡 DROWSY (Fatigue Warning)</span>'
    elif face_detected:
        status_html = '<span class="badge-attentive">🟢 ATTENTIVE (Driver Alert)</span>'
    else:
        status_html = '<span class="badge-searching">⚪ SEARCHING (No Face Detected)</span>'
    col1.markdown(f'<div class="metric-card"><div class="metric-title">Driver Alertness</div><div class="metric-value">{status_html}</div></div>', unsafe_allow_html=True)

    # 2. Eye Opening Level (EAR)
    if avg_ear >= 0.22:
        eye_html = f'<div class="metric-value" style="color:#34d399;">👁️ Eyes Open <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(EAR: {avg_ear:.2f} | Normal Vision)</div></div>'
    elif face_detected:
        eye_html = f'<div class="metric-value" style="color:#ef4444;">⚠️ Eyes Closed <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(EAR: {avg_ear:.2f} | Lids Shut)</div></div>'
    else:
        eye_html = '<div class="metric-value" style="color:#94a3b8;">-- <div style="font-size:0.8rem;color:#64748b;">(Scanning)</div></div>'
    col2.markdown(f'<div class="metric-card"><div class="metric-title">Eye Opening Level</div>{eye_html}</div>', unsafe_allow_html=True)

    # 3. Yawn / Mouth State (MAR)
    if mar >= 0.60:
        mouth_html = f'<div class="metric-value" style="color:#f59e0b;">🥱 Yawn Detected <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(MAR: {mar:.2f} | Fatigue)</div></div>'
    elif face_detected:
        mouth_html = f'<div class="metric-value" style="color:#34d399;">👄 Mouth Normal <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(MAR: {mar:.2f} | Closed/Talking)</div></div>'
    else:
        mouth_html = '<div class="metric-value" style="color:#94a3b8;">-- <div style="font-size:0.8rem;color:#64748b;">(Scanning)</div></div>'
    col3.markdown(f'<div class="metric-card"><div class="metric-title">Mouth / Yawn State</div>{mouth_html}</div>', unsafe_allow_html=True)

    # 4. Head Position (Pitch)
    if pitch < -20.0:
        head_html = f'<div class="metric-value" style="color:#ef4444;">⚠️ Nodding Off <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(Pitch: {pitch:+.1f}° | Slumped Down)</div></div>'
    elif face_detected:
        head_html = f'<div class="metric-value" style="color:#34d399;">👤 Facing Forward <div style="font-size:0.8rem;color:#94a3b8;font-weight:400;">(Pitch: {pitch:+.1f}° | Road Focus)</div></div>'
    else:
        head_html = '<div class="metric-value" style="color:#94a3b8;">-- <div style="font-size:0.8rem;color:#64748b;">(Scanning)</div></div>'
    col4.markdown(f'<div class="metric-card"><div class="metric-title">Head Orientation</div>{head_html}</div>', unsafe_allow_html=True)


# =============================================================================
# Preset Scenarios Configuration
# =============================================================================
PRESETS_DIR = _root_dir / "data" / "sample_presets"

PRESET_SCENARIOS = {
    "🟢 Active & Alert Driver (Normal Driving)": {
        "video": str(PRESETS_DIR / "driver_alert.mp4"),
        "image": str(PRESETS_DIR / "driver_alert_1.jpg"),
        "description": "Cabin camera footage of an attentive driver with wide open eyes (EAR ~0.35-0.45) looking straight ahead at the road, exhibiting natural, healthy blinks (<0.20s).",
        "expected_status": "🟢 ATTENTIVE (Driver Alert)",
        "expected_ear": "Open (EAR > 0.30)",
        "expected_alert": "✓ Normal Driving (No Alarm)"
    },
    "🟡 Driver Yawning & Fatigue (Early Warning)": {
        "video": str(PRESETS_DIR / "driver_yawn.mp4"),
        "image": str(PRESETS_DIR / "driver_yawn_1.jpg"),
        "description": "Cabin camera capturing the driver repeatedly yawning with wide vertical mouth separation (MAR > 0.65). Triggers early fatigue warning badges before sleep occurs.",
        "expected_status": "🟡 DROWSY (Fatigue Warning)",
        "expected_ear": "Open with intermittent squint",
        "expected_alert": "🟡 Yawning Warning (Amber Alert)"
    },
    "🚨 Driver Microsleep & Nodding Off (Critical Siren Alarm)": {
        "video": str(PRESETS_DIR / "driver_microsleep.mp4"),
        "image": str(PRESETS_DIR / "driver_microsleep_1.jpg"),
        "description": "Dangerous highway scenario where the driver's eyelids close and remain shut for >= 1.5 seconds while the head slumps downward. Triggers immediate Crimson Red siren and emergency audio alarm.",
        "expected_status": "🚨 CRITICAL (Microsleep Alarm!)",
        "expected_ear": "Closed (EAR < 0.15)",
        "expected_alert": "🚨 EMERGENCY SIREN TRIGGERED"
    }
}


# =============================================================================
# Sidebar Controls & Scenario Configuration
# =============================================================================
with st.sidebar:
    st.markdown("### 🚘 Perception Settings")
    media_mode = st.radio(
        "Select Ingestion Source:",
        ["Preset Driving Demos", "Live Webcam Feed", "Upload Custom Media"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎛️ Driver Assistance Settings")

    ear_thresh = st.slider(
        "🎯 Eye Closure Sensitivity (Lower = Deep Sleep Only | Higher = Catch Drowsy Squints)",
        min_value=0.15,
        max_value=0.35,
        value=0.22,
        step=0.01,
        help="Controls the Eye Aspect Ratio threshold. Values < 0.22 mean eyelids are shut."
    )

    consec_seconds = st.slider(
        "⏱️ Sleep Alarm Trigger Delay (Short: 1.0s | Normal: 1.5s | Relaxed: 2.5s)",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.25,
        help="How long eyes must stay closed before sounding the alarm (1.5s = 45 frames at 30 FPS)."
    )
    consec_frames_limit = int(consec_seconds * 30.0)

    mar_thresh = st.slider(
        "🥱 Yawn Sensitivity (Lower = Deep Yawns Only | Higher = Detect Small Yawns)",
        min_value=0.50,
        max_value=0.85,
        value=0.65,
        step=0.05,
        help="Higher values trigger only on wide yawns; lower values catch early mouth opening."
    )

    pitch_thresh = st.slider(
        "📐 Head Nodding Sensitivity (Strict: 15° Down | Normal: 25° Down | Relaxed: 35° Down)",
        min_value=15.0,
        max_value=40.0,
        value=25.0,
        step=5.0,
        help="Controls downward head slump angle before triggering a nodding-off alarm."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Video Stream Tuning")
    max_frames_to_process = st.slider(
        "⏱️ Video Clip Length (Short 2s: 60 | Standard 4s: 120 | Long 10s: 300)",
        min_value=30,
        max_value=300,
        value=120,
        step=30
    )

    frame_step = st.slider(
        "⏩ Video Frame Speed (1 = Highest Precision | 2 = Faster 2x Playback)",
        min_value=1,
        max_value=3,
        value=1,
        step=1
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="background:#1e293b;padding:12px;border-radius:8px;border-left:4px solid #8b5cf6;margin-bottom:8px;">
            <div style="color:#c084fc;font-weight:700;font-size:0.85rem;">🤖 MASTER_SDLC_AGENTS</div>
            <div style="color:#94a3b8;font-size:0.75rem;margin-top:2px;">Engineered via 7-Agent Master Multi-Agent Framework</div>
            <div style="color:#34d399;font-size:0.72rem;margin-top:4px;font-weight:600;">✓ 7/7 HITL Quality Gates Certified</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("DrowsyGuard DMS Engine • MediaPipe 3D Mesh • OpenCV 4.9+")


# =============================================================================
# Main Application Dashboard
# =============================================================================
st.title("🚗 Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection")
st.markdown(
    "Automated 3D facial landmark mesh tracking, **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, "
    "and **3D Head Pose** automotive telemetry.  \n"
    "🏛️ **Autonomous Governance:** Engineered end-to-end via the **`MASTER_SDLC_AGENTS`** framework (7 Autonomous Agents with strict Human-in-the-Loop verification)."
)

# Initialize Detector (no static cache to avoid stale state retention)
try:
    detector = DriverDrowsinessDetector(
        ear_threshold=ear_thresh,
        consec_frames=consec_frames_limit,
        mar_threshold=mar_thresh,
        pitch_threshold=pitch_thresh
    )
except Exception as e:
    st.error(
        f"### ⚠️ Perception Engine Initialization Notice\n\n"
        f"**{str(e)}**\n\n"
        f"*Resolution*: In your Streamlit Cloud dashboard, go to **Settings** (or redeploy with **Advanced Settings**) and select **Python 3.10** or **Python 3.11**."
    )
    st.stop()

# Pre-generate Audio Siren URI
siren_audio_uri = generate_audio_alert_b64(frequency_hz=880, duration_s=0.6)


# =============================================================================
# Ingestion Mode 1: Preset Driving Demos
# =============================================================================
if media_mode == "Preset Driving Demos":
    preset_choice = st.selectbox(
        "Choose Driving Scenario:",
        list(PRESET_SCENARIOS.keys())
    )

    scenario_info = PRESET_SCENARIOS[preset_choice]

    st.info(
        f"**Scenario Context:** {scenario_info['description']}\n\n"
        f"• **Expected Status:** `{scenario_info['expected_status']}`  \n"
        f"• **Expected Eye Opening:** `{scenario_info['expected_ear']}`  \n"
        f"• **Expected Alert Action:** `{scenario_info['expected_alert']}`"
    )

    test_type = st.radio(
        "Select Demo Format:",
        ["🎥 Stream Cabin Video Clip", "📸 Inspect Single HD Cabin Photo"],
        horizontal=True
    )

    if test_type == "📸 Inspect Single HD Cabin Photo":
        img_path = scenario_info["image"]
        if os.path.exists(img_path):
            raw_img = load_and_validate_image(img_path)
            detector.reset()
            result = detector.process_frame(
                raw_img,
                ear_threshold=ear_thresh,
                consec_frames=1,
                mar_threshold=mar_thresh
            )

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            render_metric_cards(result, col1, col2, col3, col4)

            # Annotated image
            annotated = draw_drowsiness_overlay(
                frame=raw_img,
                landmarks=result["landmarks_px"],
                telemetry=result
            )
            st.image(annotated, channels="RGB", use_container_width=True)

            if result["microsleep_alert"]:
                st.markdown(f'<audio autoplay><source src="{siren_audio_uri}" type="audio/wav"></audio>', unsafe_allow_html=True)
                st.error("🚨 EMERGENCY SIREN ACTIVE: The driver's eyes are shut! Wake up!")

    else:
        # Video stream
        video_path = scenario_info["video"]
        if os.path.exists(video_path):
            start_video = st.button("▶ Start Driver Monitoring Stream", type="primary")

            if start_video:
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                status_box = kpi_col1.empty()
                eye_box = kpi_col2.empty()
                mouth_box = kpi_col3.empty()
                head_box = kpi_col4.empty()

                video_placeholder = st.empty()
                audio_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                telemetry_history: List[Dict[str, Any]] = []

                detector.reset()
                frame_idx = 0
                frame_gen = extract_video_frame_iterator(video_path, max_frames=max_frames_to_process)

                for raw_frame in frame_gen:
                    frame_idx += 1
                    if frame_idx % frame_step != 0:
                        continue

                    result = detector.process_frame(
                        raw_frame,
                        ear_threshold=ear_thresh,
                        consec_frames=consec_frames_limit,
                        mar_threshold=mar_thresh
                    )
                    telemetry_history.append(result)

                    annotated = draw_drowsiness_overlay(
                        frame=raw_frame,
                        landmarks=result["landmarks_px"],
                        telemetry=result
                    )

                    # Update live KPI metric cards
                    render_metric_cards(result, status_box, eye_box, mouth_box, head_box)

                    # Render frame
                    video_placeholder.image(annotated, channels="RGB", use_container_width=True)

                    # Play Audio Siren if Microsleep triggered
                    if result["microsleep_alert"]:
                        audio_placeholder.markdown(
                            f'<audio autoplay><source src="{siren_audio_uri}" type="audio/wav"></audio>',
                            unsafe_allow_html=True
                        )

                    progress_bar.progress(min(1.0, frame_idx / max_frames_to_process))

                progress_bar.empty()
                st.success(f"✓ Video processing complete: Processed {len(telemetry_history)} frames.")

                # CSV Export
                csv_data = format_telemetry_csv(telemetry_history)
                st.download_button(
                    label="📥 Download Driver Telemetry Log (CSV)",
                    data=csv_data,
                    file_name="driver_drowsiness_log.csv",
                    mime="text/csv",
                    type="primary"
                )


# =============================================================================
# Ingestion Mode 2: Live Webcam Feed
# =============================================================================
elif media_mode == "Live Webcam Feed":
    st.subheader("📹 Live Camera Driver Monitoring")
    st.markdown("Point your webcam at your face. Blink naturally, close your eyes for 2 seconds, or yawn to test the real-time AI perception.")

    camera_photo = st.camera_input("Take a live driver snapshot or use streaming verification:")

    if camera_photo is not None:
        raw_img = load_and_validate_image(camera_photo.getvalue())
        detector.reset()
        result = detector.process_frame(
            raw_img,
            ear_threshold=ear_thresh,
            consec_frames=1,
            mar_threshold=mar_thresh
        )

        col1, col2, col3, col4 = st.columns(4)
        render_metric_cards(result, col1, col2, col3, col4)

        annotated = draw_drowsiness_overlay(
            frame=raw_img,
            landmarks=result["landmarks_px"],
            telemetry=result
        )
        st.image(annotated, channels="RGB", use_container_width=True)

        if result["microsleep_alert"]:
            st.markdown(f'<audio autoplay><source src="{siren_audio_uri}" type="audio/wav"></audio>', unsafe_allow_html=True)
            st.error("🚨 EMERGENCY SIREN ACTIVE: The driver's eyes are shut! Wake up!")


# =============================================================================
# Ingestion Mode 3: Upload Custom Media
# =============================================================================
else:
    st.subheader("📤 Upload Custom Driver Media")
    uploaded_file = st.file_uploader(
        "Upload driver cabin photo or video clip (.mp4, .avi, .jpg, .png):",
        type=["mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        is_valid, media_type, msg = validate_media_input(file_bytes, uploaded_file.name)

        if not is_valid:
            st.error(f"❌ Ingestion Error: {msg}")
        else:
            st.success(f"✓ Validated: {msg}")

            if media_type == "image":
                raw_img = load_and_validate_image(file_bytes)
                detector.reset()
                result = detector.process_frame(
                    raw_img,
                    ear_threshold=ear_thresh,
                    consec_frames=1,
                    mar_threshold=mar_thresh
                )

                col1, col2, col3, col4 = st.columns(4)
                render_metric_cards(result, col1, col2, col3, col4)

                annotated = draw_drowsiness_overlay(
                    frame=raw_img,
                    landmarks=result["landmarks_px"],
                    telemetry=result
                )
                st.image(annotated, channels="RGB", use_container_width=True)

                if result["microsleep_alert"]:
                    st.markdown(f'<audio autoplay><source src="{siren_audio_uri}" type="audio/wav"></audio>', unsafe_allow_html=True)
                    st.error("🚨 EMERGENCY SIREN ACTIVE!")

            else:
                # Video file
                start_video = st.button("▶ Run Detection on Uploaded Video", type="primary")

                if start_video:
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    status_box = kpi_col1.empty()
                    eye_box = kpi_col2.empty()
                    mouth_box = kpi_col3.empty()
                    head_box = kpi_col4.empty()

                    video_placeholder = st.empty()
                    audio_placeholder = st.empty()
                    progress_bar = st.progress(0.0)
                    telemetry_history: List[Dict[str, Any]] = []

                    detector.reset()
                    frame_idx = 0
                    frame_gen = extract_video_frame_iterator(file_bytes, max_frames=max_frames_to_process)

                    for raw_frame in frame_gen:
                        frame_idx += 1
                        if frame_idx % frame_step != 0:
                            continue

                        result = detector.process_frame(
                            raw_frame,
                            ear_threshold=ear_thresh,
                            consec_frames=consec_frames_limit,
                            mar_threshold=mar_thresh
                        )
                        telemetry_history.append(result)

                        annotated = draw_drowsiness_overlay(
                            frame=raw_frame,
                            landmarks=result["landmarks_px"],
                            telemetry=result
                        )

                        render_metric_cards(result, status_box, eye_box, mouth_box, head_box)
                        video_placeholder.image(annotated, channels="RGB", use_container_width=True)

                        if result["microsleep_alert"]:
                            audio_placeholder.markdown(
                                f'<audio autoplay><source src="{siren_audio_uri}" type="audio/wav"></audio>',
                                unsafe_allow_html=True
                            )

                        progress_bar.progress(min(1.0, frame_idx / max_frames_to_process))

                    progress_bar.empty()
                    st.success(f"✓ Video processing complete: Processed {len(telemetry_history)} frames.")

                    csv_data = format_telemetry_csv(telemetry_history)
                    st.download_button(
                        label="📥 Download Telemetry Log (CSV)",
                        data=csv_data,
                        file_name="custom_driver_log.csv",
                        mime="text/csv",
                        type="primary"
                    )


# =============================================================================
# Autonomous SDLC Framework Footer Attribution
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:0.82rem;padding:12px 0 24px 0;">
        🏛️ <b>Autonomous SDLC Framework:</b> Engineered end-to-end via <code>MASTER_SDLC_AGENTS</code><br>
        (Product Manager • Technical Architect • Tech Lead • Software Engineer • QA Engineer • Security Auditor • DevOps Engineer)<br>
        <span style="color:#10b981;">✓ 3D Facial Mesh Engine</span> • <span style="color:#3b82f6;">✓ Real-Time EAR / MAR ADAS Telemetry</span> • <span style="color:#a855f7;">✓ Production Ready</span>
    </div>
    """,
    unsafe_allow_html=True
)
