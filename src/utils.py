"""
Utility, Image Processing & Painter HUD Annotation Module
Project: Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System
System Codename: DrowsyGuard DMS & Automotive Telemetry Engine
"""

import os
import io
import wave
import struct
import base64
import tempfile
from typing import Union, List, Dict, Tuple, Optional, Generator, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2

# =============================================================================
# Defensive Security: Cap Max Pixels to Prevent Decompression Bomb DoS Attacks
# =============================================================================
Image.MAX_IMAGE_PIXELS = 50_000_000

# =============================================================================
# Visual Contract Constants (RGB Color Format)
# =============================================================================
COLOR_ATTENTIVE: Tuple[int, int, int] = (34, 197, 94)      # Emerald Green
COLOR_DROWSY: Tuple[int, int, int] = (245, 158, 11)        # Golden Amber / Warning
COLOR_CRITICAL: Tuple[int, int, int] = (239, 68, 68)       # Crimson Red / Emergency
COLOR_EYE_CONTOUR: Tuple[int, int, int] = (0, 220, 255)    # Electric Cyan
COLOR_MOUTH_CONTOUR: Tuple[int, int, int] = (255, 180, 0)  # Bright Amber
COLOR_TEXT_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_HUD_BG: Tuple[int, int, int] = (15, 23, 42)          # Dark Slate Navy


def validate_media_input(
    file_bytes: Union[bytes, io.BytesIO],
    filename: str
) -> Tuple[bool, str, str]:
    """
    Validates uploaded file bytes to ensure they represent a valid, supported media container.

    Args:
        file_bytes: In-memory byte buffer or raw bytes.
        filename: Original file name string.

    Returns:
        tuple: (is_valid: bool, media_type: str ['image', 'video', 'unknown'], message: str)
    """
    if not file_bytes:
        return False, "unknown", "File byte stream is empty."

    raw_bytes = file_bytes.getvalue() if isinstance(file_bytes, io.BytesIO) else file_bytes
    if len(raw_bytes) == 0:
        return False, "unknown", "File payload has 0 bytes."

    ext = os.path.splitext(filename.lower())[1]

    # Image format verification
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            stream = io.BytesIO(raw_bytes)
            with Image.open(stream) as img:
                img.verify()
            return True, "image", f"Valid image format: {ext}"
        except Exception as e:
            return False, "image", f"Corrupt image payload: {str(e)}"

    # Video container verification
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        if len(raw_bytes) < 1024:
            return False, "video", "Video file payload is too small to contain valid container headers."
        return True, "video", f"Supported video container: {ext}"

    else:
        return False, "unknown", f"Unsupported file extension: {ext}. Expected .mp4, .avi, .jpg, .png."


def load_and_validate_image(
    image_input: Union[bytes, io.BytesIO, Image.Image, np.ndarray, str]
) -> np.ndarray:
    """
    Ingests file bytes, BytesIO stream, PIL Image, file path, or NumPy array.
    Ensures standard 3-channel RGB uint8 format with shape (H, W, 3).

    Returns:
        np.ndarray: An image array of shape (H, W, 3) and dtype uint8.

    Raises:
        ValueError: If input is corrupt, empty, or cannot be parsed as an RGB image.
    """
    if image_input is None:
        raise ValueError("Image input cannot be None.")

    try:
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if img.size == 0 or len(img.shape) < 2:
                raise ValueError("NumPy array is empty or has invalid dimensions.")
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] != 3:
                raise ValueError(f"Unsupported number of image channels: {img.shape[2]}")
            return img.astype(np.uint8)

        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            if arr.size == 0:
                raise ValueError("Converted PIL image array is empty.")
            return arr.astype(np.uint8)

        elif isinstance(image_input, (bytes, io.BytesIO)):
            raw_bytes = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
            if len(raw_bytes) == 0:
                raise ValueError("Byte stream is empty.")
            stream = io.BytesIO(raw_bytes)
            with Image.open(stream) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                arr = np.array(pil_img)
            return arr.astype(np.uint8)

        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Image path does not exist: {image_input}")
            with Image.open(image_input) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                arr = np.array(pil_img)
            return arr.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to decode image input: {str(e)}") from e


def extract_video_frame_iterator(
    video_source: Union[str, bytes],
    max_frames: int = 450
) -> Generator[np.ndarray, None, None]:
    """
    Decodes video frames from a file path or in-memory byte buffer.
    Yields standardized RGB uint8 frames sequentially with bounded memory isolation.

    Args:
        video_source: File path string or raw video bytes.
        max_frames: Maximum number of frames to yield (prevents unbounded memory consumption).

    Yields:
        np.ndarray: Sequential video frame of shape (H, W, 3) in RGB format.
    """
    temp_file_path = None
    try:
        if isinstance(video_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(video_source)
                temp_file_path = tf.name
            target_path = temp_file_path
        else:
            target_path = video_source

        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video stream from source: {target_path}")

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield rgb_frame
            frame_count += 1

        cap.release()

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def draw_drowsiness_overlay(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    telemetry: Optional[Dict[str, Any]] = None,
    show_mesh: bool = True
) -> np.ndarray:
    """
    The Painter: Composites the facial landmark contours, dynamic driver bounding box,
    on-screen Heads-Up Display (HUD), and emergency flashing sirens directly onto the frame.

    Args:
        frame: Original RGB video frame of shape (H, W, 3).
        landmarks: Normalized or pixel coordinates array (N, 2) or (N, 3).
        telemetry: Dictionary containing ear, mar, driver_status, alert flags, and head pitch.
        show_mesh: Boolean to toggle fine landmark mesh dots on eyes and lips.

    Returns:
        np.ndarray: Annotated RGB video frame.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot render overlay on an empty or None frame.")

    canvas = frame.copy()
    h, w = canvas.shape[:2]

    status = "ATTENTIVE"
    avg_ear = 0.0
    mar = 0.0
    pitch_deg = 0.0
    is_microsleep = False
    is_yawn = False
    face_detected = False

    if telemetry is not None:
        status = telemetry.get("driver_status", "ATTENTIVE")
        avg_ear = float(telemetry.get("avg_ear", 0.0))
        mar = float(telemetry.get("mar", 0.0))
        pitch_deg = float(telemetry.get("head_pitch_deg", 0.0))
        is_microsleep = bool(telemetry.get("microsleep_alert", False))
        is_yawn = bool(telemetry.get("yawn_alert", False))
        face_detected = bool(telemetry.get("face_detected", False))

    # Resolve Status Color
    if is_microsleep or status == "CRITICAL":
        status_color = COLOR_CRITICAL
        badge_text = "🚨 CRITICAL: MICROSLEEP ALARM!"
    elif is_yawn or status == "DROWSY":
        status_color = COLOR_DROWSY
        badge_text = "🟡 WARNING: DROWSINESS / YAWNING"
    elif face_detected:
        status_color = COLOR_ATTENTIVE
        badge_text = "🟢 ATTENTIVE: DRIVER ALERT"
    else:
        status_color = (148, 163, 184) # Gray
        badge_text = "⚪ SEARCHING: DRIVER NOT IN VIEW"

    # -------------------------------------------------------------------------
    # 1. Draw Landmark Contours & Face Bounding Box
    # -------------------------------------------------------------------------
    has_bbox = False
    min_x = min_y = max_x = max_y = 0

    if telemetry and "face_bbox" in telemetry and telemetry["face_bbox"] and telemetry["face_bbox"] != [0, 0, 0, 0]:
        min_x, min_y, max_x, max_y = telemetry["face_bbox"]
        has_bbox = True
    elif telemetry and "all_landmarks_px" in telemetry and len(telemetry["all_landmarks_px"]) >= 468:
        all_lms = telemetry["all_landmarks_px"]
        raw_min_x = int(np.min(all_lms[:, 0]))
        raw_max_x = int(np.max(all_lms[:, 0]))
        raw_min_y = int(np.min(all_lms[:, 1]))
        raw_max_y = int(np.max(all_lms[:, 1]))
        bw = raw_max_x - raw_min_x
        bh = raw_max_y - raw_min_y
        min_x = max(0, raw_min_x - int(0.12 * bw))
        max_x = min(w - 1, raw_max_x + int(0.12 * bw))
        min_y = max(0, raw_min_y - int(0.20 * bh))
        max_y = min(h - 1, raw_max_y + int(0.10 * bh))
        has_bbox = True
    elif landmarks is not None and len(landmarks) > 0:
        lm_px = landmarks.astype(np.int32)
        raw_min_x = int(np.min(lm_px[:, 0]))
        raw_max_x = int(np.max(lm_px[:, 0]))
        raw_min_y = int(np.min(lm_px[:, 1]))
        raw_max_y = int(np.max(lm_px[:, 1]))
        bw = max(10, raw_max_x - raw_min_x)
        bh = max(10, raw_max_y - raw_min_y)
        # Proportionately frame full driver head
        min_x = max(0, raw_min_x - int(0.35 * bw))
        max_x = min(w - 1, raw_max_x + int(0.35 * bw))
        min_y = max(0, raw_min_y - int(0.55 * bh))
        max_y = min(h - 1, raw_max_y + int(0.25 * bh))
        has_bbox = True

    if has_bbox and (max_x > min_x) and (max_y > min_y):
        # Draw Dynamic Driver Bounding Box
        stroke = max(2, int(w * 0.003))
        cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), status_color, stroke)

        # Corner Accents on Bounding Box
        corner_len = min(25, (max_x - min_x) // 5)
        cv2.line(canvas, (min_x, min_y), (min_x + corner_len, min_y), status_color, stroke + 2)
        cv2.line(canvas, (min_x, min_y), (min_x, min_y + corner_len), status_color, stroke + 2)
        cv2.line(canvas, (max_x, min_y), (max_x - corner_len, min_y), status_color, stroke + 2)
        cv2.line(canvas, (max_x, min_y), (max_x, min_y + corner_len), status_color, stroke + 2)
        cv2.line(canvas, (min_x, max_y), (min_x + corner_len, max_y), status_color, stroke + 2)
        cv2.line(canvas, (min_x, max_y), (min_x, max_y - corner_len), status_color, stroke + 2)
        cv2.line(canvas, (max_x, max_y), (max_x - corner_len, max_y), status_color, stroke + 2)
        cv2.line(canvas, (max_x, max_y), (max_x, max_y - corner_len), status_color, stroke + 2)

    # Draw Eye and Mouth Contours if mesh enabled
    if show_mesh and landmarks is not None and len(landmarks) >= 20:
        lm_px = landmarks.astype(np.int32)
        # Draw eye contour points in Electric Cyan
        for pt in lm_px[:12]:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, COLOR_EYE_CONTOUR, -1, lineType=cv2.LINE_AA)
        # Draw mouth contour points in Bright Amber
        for pt in lm_px[12:20]:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, COLOR_MOUTH_CONTOUR, -1, lineType=cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # 2. Emergency Flashing Top Siren Banner on Critical Microsleep
    # -------------------------------------------------------------------------
    if is_microsleep:
        banner_h = max(36, int(h * 0.08))
        banner_overlay = canvas.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), COLOR_CRITICAL, -1)
        canvas = cv2.addWeighted(banner_overlay, 0.85, canvas, 0.15, 0)
        siren_font_scale = max(0.55, min(0.95, w / 1200.0))
        cv2.putText(
            canvas,
            "WAKE UP! MICROSLEEP DETECTED!",
            (w // 2 - int(w * 0.28), int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            siren_font_scale,
            COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA
        )

    # -------------------------------------------------------------------------
    # 3. Heads-Up Display (HUD) Telemetry Box (Top-Left)
    # -------------------------------------------------------------------------
    hud_w = max(220, min(380, int(w * 0.38)))
    hud_h = max(80, min(120, int(h * 0.22)))
    hud_top = max(12, int(h * 0.09) if is_microsleep else 12)

    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (12, hud_top), (12 + hud_w, hud_top + hud_h), COLOR_HUD_BG, -1)
    canvas = cv2.addWeighted(hud_overlay, 0.78, canvas, 0.22, 0)
    cv2.rectangle(canvas, (12, hud_top), (12 + hud_w, hud_top + hud_h), (51, 65, 85), 1, cv2.LINE_AA)

    font_scale = max(0.38, min(0.52, w / 2200.0))
    font = cv2.FONT_HERSHEY_SIMPLEX

    eye_str = f"Eyes: {'OPEN' if avg_ear >= 0.22 else 'CLOSED'} (EAR: {avg_ear:.2f})"
    mouth_str = f"Mouth: {'YAWNING' if mar >= 0.60 else 'NORMAL'} (MAR: {mar:.2f})"
    pitch_str = f"Head: {'SLUMPED' if pitch_deg < -20.0 else 'FORWARD'} ({pitch_deg:+.1f} deg)"

    y_line1 = int(hud_top + hud_h * 0.25)
    y_line2 = int(hud_top + hud_h * 0.48)
    y_line3 = int(hud_top + hud_h * 0.70)
    y_badge = int(hud_top + hud_h * 0.94)

    cv2.putText(canvas, eye_str, (22, y_line1), font, font_scale, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, mouth_str, (22, y_line2), font, font_scale, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, pitch_str, (22, y_line3), font, font_scale, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    # Status Pill on HUD Box
    badge_rect_top = int(hud_top + hud_h * 0.76)
    cv2.rectangle(canvas, (22, badge_rect_top), (22 + min(320, hud_w - 20), y_badge), status_color, -1)
    cv2.putText(canvas, badge_text, (28, int(y_badge - 4)), font, font_scale * 0.85, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    return canvas


def generate_audio_alert_b64(frequency_hz: int = 880, duration_s: float = 0.5) -> str:
    """
    Synthesizes an in-memory high-frequency warning tone (WAV audio format)
    and encodes it to a base64 Data URI string. Allows the browser to sound
    instant acoustic alarms without external MP3 asset dependencies.

    Args:
        frequency_hz: Frequency in Hertz (default: 880 Hz, concert A5).
        duration_s: Duration in seconds.

    Returns:
        str: Base64 data URI string ("data:audio/wav;base64,...").
    """
    sample_rate = 22050
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)

        # Generate sine wave with fade-out envelope to avoid audio clicks
        for i in range(num_samples):
            t = float(i) / sample_rate
            decay = max(0.0, 1.0 - (float(i) / num_samples) * 0.1)
            sample_val = int(32767.0 * 0.45 * decay * np.sin(2.0 * np.pi * frequency_hz * t))
            data = struct.pack("<h", sample_val)
            wf.writeframesraw(data)

    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def format_telemetry_csv(records: List[Dict[str, Any]]) -> str:
    """
    Serializes per-frame driver monitoring telemetry records into standard
    RFC 4180 CSV format for 1-click download and fleet auditing.
    Hardened against spreadsheet formula injection.

    Args:
        records: List of frame telemetry dictionaries.

    Returns:
        str: Valid CSV string.
    """
    columns = [
        "Frame #",
        "Timestamp (s)",
        "Driver Status",
        "Average EAR",
        "Mouth MAR",
        "Head Pitch (deg)",
        "Microsleep Alert",
        "Yawn Alert",
        "Head Slump Alert",
        "Inference Latency (ms)",
    ]

    if not records:
        df = pd.DataFrame(columns=columns)
        return df.to_csv(index=False)

    rows = []
    for r in records:
        f_id = int(r.get("frame_id", 0))
        ts = round(float(r.get("timestamp_s", 0.0)), 3)
        status = str(r.get("driver_status", "ATTENTIVE")).replace("=", "").replace("+", "").replace("-", "")
        ear = round(float(r.get("avg_ear", 0.0)), 3)
        mar = round(float(r.get("mar", 0.0)), 3)
        pitch = round(float(r.get("head_pitch_deg", 0.0)), 1)
        ms_alert = "YES" if r.get("microsleep_alert", False) else "NO"
        yawn_alert = "YES" if r.get("yawn_alert", False) else "NO"
        slump_alert = "YES" if r.get("head_slump_alert", False) else "NO"
        latency = round(float(r.get("inference_latency_ms", 0.0)), 1)

        rows.append(
            {
                "Frame #": f_id,
                "Timestamp (s)": ts,
                "Driver Status": status,
                "Average EAR": ear,
                "Mouth MAR": mar,
                "Head Pitch (deg)": pitch,
                "Microsleep Alert": ms_alert,
                "Yawn Alert": yawn_alert,
                "Head Slump Alert": slump_alert,
                "Inference Latency (ms)": latency,
            }
        )

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
