"""
Unit Test Suite for Utility & Painter HUD Functions (tests/test_utils.py).
Project: Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System
Covers: Media validation, image decoding, overlay rendering, audio siren synthesis, CSV logging.
"""

import io
import base64
import numpy as np
import pytest
from PIL import Image
import cv2

from src.utils import (
    validate_media_input,
    load_and_validate_image,
    extract_video_frame_iterator,
    draw_drowsiness_overlay,
    generate_audio_alert_b64,
    format_telemetry_csv
)


class TestMediaValidation:
    """Tests for validate_media_input function."""

    def test_valid_jpeg(self, valid_jpg_bytes: bytes):
        ok, media_type, msg = validate_media_input(valid_jpg_bytes, "test_driver.jpg")
        assert ok is True
        assert media_type == "image"
        assert "Valid image" in msg

    def test_valid_png(self, valid_png_bytes: bytes):
        ok, media_type, msg = validate_media_input(valid_png_bytes, "test_driver.png")
        assert ok is True
        assert media_type == "image"
        assert "Valid image" in msg

    def test_empty_file_rejected(self):
        ok, media_type, msg = validate_media_input(b"", "empty.jpg")
        assert ok is False
        assert media_type == "unknown"
        assert "empty" in msg.lower() or "0 bytes" in msg.lower()

    def test_unsupported_extension_rejected(self, valid_jpg_bytes: bytes):
        ok, media_type, msg = validate_media_input(valid_jpg_bytes, "payload.exe")
        assert ok is False
        assert media_type == "unknown"
        assert "unsupported" in msg.lower()

    def test_corrupted_header_rejected(self, corrupted_bytes: bytes):
        ok, media_type, msg = validate_media_input(corrupted_bytes, "corrupt.jpg")
        assert ok is False
        assert media_type == "image"
        assert "corrupt" in msg.lower()

    def test_video_payload_too_small_rejected(self):
        tiny_video = b"ftyp" + b"\x00" * 10
        ok, media_type, msg = validate_media_input(tiny_video, "drive.mp4")
        assert ok is False
        assert media_type == "video"
        assert "too small" in msg.lower()

    def test_valid_mp4_header(self):
        # ftyp box magic header for MP4 (>= 1024 bytes)
        mp4_payload = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024
        ok, media_type, msg = validate_media_input(mp4_payload, "drive.mp4")
        assert ok is True
        assert media_type == "video"


class TestImageLoadingAndNormalization:
    """Tests for load_and_validate_image function."""

    def test_load_from_valid_bytes(self, valid_jpg_bytes: bytes):
        img = load_and_validate_image(valid_jpg_bytes)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_from_pil_image(self):
        pil_img = Image.new("RGB", (120, 80), color=(255, 0, 0))
        img = load_and_validate_image(pil_img)
        assert isinstance(img, np.ndarray)
        assert img.shape == (80, 120, 3)
        assert img[40, 60, 0] == 255  # Red channel preserved

    def test_load_from_numpy_array(self):
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img = load_and_validate_image(arr)
        assert img.shape == (100, 100, 3)

    def test_load_rgba_strips_alpha(self):
        rgba = Image.new("RGBA", (64, 64), color=(0, 255, 0, 128))
        img = load_and_validate_image(rgba)
        assert img.shape == (64, 64, 3)

    def test_reject_corrupted_bytes(self, corrupted_bytes: bytes):
        with pytest.raises(ValueError, match="Failed to decode image"):
            load_and_validate_image(corrupted_bytes)

    def test_reject_undersized_image(self):
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty or has invalid dimensions"):
            load_and_validate_image(empty_arr)


class TestAudioAlertGeneration:
    """Tests for generate_audio_alert_b64 siren synthesis."""

    def test_siren_uri_format(self):
        uri = generate_audio_alert_b64(frequency_hz=880, duration_s=0.2)
        assert uri.startswith("data:audio/wav;base64,")

    def test_siren_wav_header_validity(self):
        uri = generate_audio_alert_b64(frequency_hz=440, duration_s=0.1)
        b64_str = uri.split(",")[1]
        wav_bytes = base64.b64decode(b64_str)
        # Standard WAV begins with 'RIFF' and has 'WAVE' at byte 8
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"


class TestCSVTelemetrySerialization:
    """Tests for format_telemetry_csv function."""

    def test_empty_records_returns_header(self):
        csv_out = format_telemetry_csv([])
        assert "Driver Status" in csv_out
        assert "Average EAR" in csv_out
        lines = csv_out.strip().splitlines()
        assert len(lines) == 1

    def test_valid_records_serialization(self):
        records = [
            {
                "frame_id": 1,
                "timestamp_s": 0.033,
                "face_detected": True,
                "driver_status": "ATTENTIVE",
                "left_ear": 0.32,
                "right_ear": 0.31,
                "avg_ear": 0.315,
                "mar": 0.35,
                "head_pitch_deg": 2.1,
                "head_yaw_deg": -1.4,
                "head_roll_deg": 0.5,
                "total_blinks": 0,
                "latency_ms": 15.2
            }
        ]
        csv_out = format_telemetry_csv(records)
        assert "ATTENTIVE" in csv_out
        assert "0.315" in csv_out
        lines = csv_out.strip().splitlines()
        assert len(lines) == 2

    def test_formula_injection_protection(self):
        malicious = [
            {
                "frame_id": 1,
                "driver_status": "=cmd|' /C calc'!A0",
                "face_detected": True
            }
        ]
        csv_out = format_telemetry_csv(malicious)
        # Leading dangerous formula triggers (=, +, -) must be stripped
        assert "=cmd|" not in csv_out
        assert "cmd|' /C calc'!A0" in csv_out


class TestOverlayRendering:
    """Tests for draw_drowsiness_overlay graphics compositing."""

    def test_overlay_preserves_dimensions(self, blank_frame: np.ndarray):
        telemetry = {
            "face_detected": False,
            "driver_status": "SEARCHING",
            "avg_ear": 0.0,
            "mar": 0.0,
            "head_pitch_deg": 0.0,
            "latency_ms": 10.0
        }
        out = draw_drowsiness_overlay(blank_frame, np.empty((0, 2), dtype=np.int32), telemetry)
        assert out.shape == blank_frame.shape
        assert out.dtype == np.uint8

    def test_overlay_rendering_with_landmarks(self, blank_frame: np.ndarray):
        # Generate dummy 468 landmark coordinates
        landmarks = np.zeros((468, 2), dtype=np.int32)
        landmarks[:, 0] = 320
        landmarks[:, 1] = 240
        telemetry = {
            "face_detected": True,
            "driver_status": "ATTENTIVE",
            "avg_ear": 0.32,
            "mar": 0.35,
            "head_pitch_deg": 5.0,
            "latency_ms": 18.5
        }
        out = draw_drowsiness_overlay(blank_frame, landmarks, telemetry)
        assert out.shape == (480, 640, 3)
        # Verify HUD modification (not identical to pure blank)
        assert not np.array_equal(out, blank_frame)
