"""
PyTest Configuration & Test Fixture Provider for Driver Drowsiness Detection System.
Provides isolated, reproducible test fixtures for Unit and Integration test suites.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import DriverDrowsinessDetector


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute Path of the project root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def presets_dir(project_root: Path) -> Path:
    """Returns the Path to sample preset media directory."""
    return project_root / "data" / "sample_presets"


@pytest.fixture(scope="function")
def detector() -> DriverDrowsinessDetector:
    """Provides a fresh DriverDrowsinessDetector instance in static image mode."""
    det = DriverDrowsinessDetector(
        ear_threshold=0.22,
        consec_frames=45,
        mar_threshold=0.65,
        pitch_threshold=25.0,
        static_image_mode=True
    )
    yield det
    det.close()


@pytest.fixture(scope="function")
def video_detector() -> DriverDrowsinessDetector:
    """Provides a fresh DriverDrowsinessDetector instance in video stream mode."""
    det = DriverDrowsinessDetector(
        ear_threshold=0.22,
        consec_frames=45,
        mar_threshold=0.65,
        pitch_threshold=25.0,
        static_image_mode=False
    )
    yield det
    det.close()


@pytest.fixture(scope="session")
def sample_alert_image(presets_dir: Path) -> np.ndarray:
    """Loads and returns the real alert driver RGB image."""
    img_path = presets_dir / "driver_alert_1.jpg"
    assert img_path.exists(), f"Missing preset: {img_path}"
    bgr = cv2.imread(str(img_path))
    assert bgr is not None, f"Failed to decode: {img_path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def sample_yawn_image(presets_dir: Path) -> np.ndarray:
    """Loads and returns the real yawning driver RGB image."""
    img_path = presets_dir / "driver_yawn_1.jpg"
    assert img_path.exists(), f"Missing preset: {img_path}"
    bgr = cv2.imread(str(img_path))
    assert bgr is not None, f"Failed to decode: {img_path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def sample_microsleep_image(presets_dir: Path) -> np.ndarray:
    """Loads and returns the real microsleep driver RGB image."""
    img_path = presets_dir / "driver_microsleep_1.jpg"
    assert img_path.exists(), f"Missing preset: {img_path}"
    bgr = cv2.imread(str(img_path))
    assert bgr is not None, f"Failed to decode: {img_path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def blank_frame() -> np.ndarray:
    """Generates a 640x480 black RGB image (no face)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def noise_frame() -> np.ndarray:
    """Generates a 640x480 uniform random noise RGB image (no face)."""
    np.random.seed(42)
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def valid_jpg_bytes() -> bytes:
    """Generates valid JPEG file bytes in memory."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def valid_png_bytes() -> bytes:
    """Generates valid PNG file bytes in memory."""
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def corrupted_bytes() -> bytes:
    """Generates corrupted non-decodable byte array."""
    return b"CORRUPTED_HEADER_DATA_NOT_AN_IMAGE_OR_VIDEO"
