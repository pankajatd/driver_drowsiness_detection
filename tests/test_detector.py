"""
Integration Test Suite for Driver Fatigue & Perception Engine (tests/test_detector.py).
Project: Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System
Covers: MediaPipe landmark extraction, EAR math, MAR math, Head Pose, Smart Stopwatch blinks/microsleeps, latency benchmark.
"""

import time
import numpy as np
import pytest

from src.detector import DriverDrowsinessDetector


class TestDetectorInitialization:
    """Verifies detector initialization, parameters, and lifecycle."""

    def test_default_initialization(self, detector: DriverDrowsinessDetector):
        assert detector.ear_threshold == 0.22
        assert detector.consec_frames == 45
        assert detector.mar_threshold == 0.65
        assert detector.pitch_threshold == 25.0
        assert detector.frame_counter == 0
        assert detector.total_blinks == 0

    def test_custom_parameters_initialization(self):
        custom_det = DriverDrowsinessDetector(
            ear_threshold=0.25,
            consec_frames=30,
            mar_threshold=0.70,
            pitch_threshold=20.0
        )
        assert custom_det.ear_threshold == 0.25
        assert custom_det.consec_frames == 30
        assert custom_det.mar_threshold == 0.70
        assert custom_det.pitch_threshold == 20.0
        custom_det.close()


class TestAspectCalculations:
    """Tests geometric mathematical formulations for EAR and MAR."""

    def test_calculate_ear_open_eye(self, detector: DriverDrowsinessDetector):
        # Synthetic open eye coordinates: horizontal width = 20, vertical height = 10
        # p1=(0,5), p2=(5,10), p3=(15,10), p4=(20,5), p5=(15,0), p6=(5,0)
        open_eye = np.array([
            [0.0, 5.0],
            [5.0, 10.0],
            [15.0, 10.0],
            [20.0, 5.0],
            [15.0, 0.0],
            [5.0, 0.0]
        ], dtype=np.float64)
        ear = detector.calculate_ear(open_eye)
        # Vertical distances: ||p2-p6|| = 10, ||p3-p5|| = 10. Horizontal: ||p1-p4|| = 20
        # EAR = (10 + 10) / (2 * 20) = 0.50
        assert pytest.approx(ear, abs=1e-3) == 0.50

    def test_calculate_ear_closed_eye(self, detector: DriverDrowsinessDetector):
        # Synthetic closed eye coordinates: horizontal width = 20, vertical height = 1
        closed_eye = np.array([
            [0.0, 5.0],
            [5.0, 5.5],
            [15.0, 5.5],
            [20.0, 5.0],
            [15.0, 4.5],
            [5.0, 4.5]
        ], dtype=np.float64)
        ear = detector.calculate_ear(closed_eye)
        # Vertical distances: 1 + 1 = 2. Horizontal: 20
        # EAR = 2 / 40 = 0.05
        assert pytest.approx(ear, abs=1e-3) == 0.05

    def test_calculate_mar_closed_mouth(self, detector: DriverDrowsinessDetector):
        # Synthetic closed mouth: width = 40, height = 4
        corners = np.array([[0.0, 10.0], [40.0, 10.0]], dtype=np.float64)
        top = np.array([[10.0, 12.0], [20.0, 12.0], [30.0, 12.0]], dtype=np.float64)
        bot = np.array([[10.0, 8.0], [20.0, 8.0], [30.0, 8.0]], dtype=np.float64)
        mar = detector.calculate_mar(corners, top, bot)
        # Sum of vertical distances = 4 + 4 + 4 = 12.0, 2 * width = 80.0 -> MAR = 12/80 = 0.15
        assert pytest.approx(mar, abs=1e-3) == 0.15

    def test_calculate_mar_yawning_mouth(self, detector: DriverDrowsinessDetector):
        # Synthetic wide open mouth: width = 40, height = 36
        corners = np.array([[0.0, 20.0], [40.0, 20.0]], dtype=np.float64)
        top = np.array([[10.0, 38.0], [20.0, 38.0], [30.0, 38.0]], dtype=np.float64)
        bot = np.array([[10.0, 2.0], [20.0, 2.0], [30.0, 2.0]], dtype=np.float64)
        mar = detector.calculate_mar(corners, top, bot)
        # Sum of vertical distances = 36 + 36 + 36 = 108.0, 2 * width = 80.0 -> MAR = 108/80 = 1.35
        assert pytest.approx(mar, abs=1e-3) == 1.35


class TestRobustnessAndResilience:
    """Verifies system stability against non-face inputs."""

    def test_blank_frame_handled_gracefully(self, detector: DriverDrowsinessDetector, blank_frame: np.ndarray):
        res = detector.process_frame(blank_frame)
        assert res["face_detected"] is False
        assert res["driver_status"] == "SEARCHING"
        assert res["avg_ear"] == 0.0
        assert res["mar"] == 0.0
        assert len(res["landmarks_px"]) == 0

    def test_noise_frame_handled_gracefully(self, detector: DriverDrowsinessDetector, noise_frame: np.ndarray):
        res = detector.process_frame(noise_frame)
        assert res["face_detected"] is False
        assert res["driver_status"] == "SEARCHING"


class TestPresetCabinBenchmarks:
    """Verifies detection accuracy on real in-cabin benchmark media."""

    def test_benchmark_alert_driver(self, detector: DriverDrowsinessDetector, sample_alert_image: np.ndarray):
        res = detector.process_frame(sample_alert_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "ATTENTIVE"
        assert res["avg_ear"] >= 0.24, f"Expected alert EAR >= 0.24, got {res['avg_ear']}"
        assert res["mar"] < 0.65, f"Expected non-yawn MAR < 0.65, got {res['mar']}"

    def test_benchmark_yawn_driver(self, detector: DriverDrowsinessDetector, sample_yawn_image: np.ndarray):
        res = detector.process_frame(sample_yawn_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "DROWSY"
        assert res["mar"] >= 0.65, f"Expected yawning MAR >= 0.65, got {res['mar']}"

    def test_benchmark_microsleep_driver(self, detector: DriverDrowsinessDetector, sample_microsleep_image: np.ndarray):
        res = detector.process_frame(sample_microsleep_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "CRITICAL"
        assert res["avg_ear"] < 0.22, f"Expected closed eyes EAR < 0.22, got {res['avg_ear']}"


class TestSmartStopwatchAndTemporalLogic:
    """Tests blink counter accumulation and microsleep emergency escalation."""

    def test_natural_blink_increments_counter(self, video_detector: DriverDrowsinessDetector, sample_alert_image: np.ndarray, sample_microsleep_image: np.ndarray):
        video_detector.reset()
        initial_blinks = video_detector.total_blinks

        # Frame 1: Alert
        video_detector.process_frame(sample_alert_image)
        # Frames 2-4: Eyes closed for 3 frames (typical 100ms blink)
        video_detector.process_frame(sample_microsleep_image)
        video_detector.process_frame(sample_microsleep_image)
        video_detector.process_frame(sample_microsleep_image)
        # Frame 5: Eyes reopen
        res = video_detector.process_frame(sample_alert_image)

        assert res["total_blinks"] == initial_blinks + 1
        assert res["driver_status"] == "ATTENTIVE"

    def test_microsleep_emergency_alarm_escalation(self, video_detector: DriverDrowsinessDetector, sample_microsleep_image: np.ndarray):
        video_detector.reset()
        alarm_triggered = False

        # Feed 46 consecutive closed-eye frames (> 45 limit)
        for _ in range(46):
            res = video_detector.process_frame(sample_microsleep_image, consec_frames=45)
            if res["microsleep_alert"] and res["driver_status"] == "CRITICAL":
                alarm_triggered = True

        assert alarm_triggered is True

    def test_reset_lifecycle(self, detector: DriverDrowsinessDetector, sample_alert_image: np.ndarray):
        detector.process_frame(sample_alert_image)
        assert detector.frame_counter > 0
        detector.reset()
        assert detector.frame_counter == 0
        assert detector.consecutive_closed_frames == 0
        assert detector.consecutive_yawn_frames == 0
        assert detector.total_blinks == 0


class TestInferenceLatency:
    """Verifies that perception engine complies with real-time latency budget (<100ms on CPU)."""

    def test_cpu_inference_latency(self, detector: DriverDrowsinessDetector, sample_alert_image: np.ndarray):
        # Warmup pass
        detector.process_frame(sample_alert_image)

        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = detector.process_frame(sample_alert_image)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

        avg_latency = np.mean(latencies)
        assert avg_latency < 100.0, f"Inference latency too slow: {avg_latency:.2f}ms"
        assert res["inference_latency_ms"] < 100.0



class TestFallbackModeResilience:
    """Verifies that OpenCV Fallback Engine operates flawlessly in zero-dependency cloud environments."""

    def test_fallback_mode_alert_detection(self, sample_alert_image: np.ndarray):
        det = DriverDrowsinessDetector(static_image_mode=True)
        det.fallback_mode = True
        det.face_mesh = None
        res = det.process_frame(sample_alert_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "ATTENTIVE"
        assert res["avg_ear"] >= 0.24
        assert res["microsleep_alert"] is False
        det.close()

    def test_fallback_mode_microsleep_alarm(self, sample_microsleep_image: np.ndarray):
        det = DriverDrowsinessDetector(static_image_mode=True)
        det.fallback_mode = True
        det.face_mesh = None
        res = det.process_frame(sample_microsleep_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "CRITICAL"
        assert res["microsleep_alert"] is True
        assert res["avg_ear"] < 0.22
        det.close()

    def test_fallback_mode_yawn_alarm(self, sample_yawn_image: np.ndarray):
        det = DriverDrowsinessDetector(static_image_mode=True)
        det.fallback_mode = True
        det.face_mesh = None
        res = det.process_frame(sample_yawn_image, consec_frames=1)
        assert res["face_detected"] is True
        assert res["driver_status"] == "DROWSY"
        assert res["yawn_alert"] is True
        assert res["mar"] >= 0.65
        det.close()

    def test_fallback_mode_blank_and_noise(self, blank_frame: np.ndarray, noise_frame: np.ndarray):
        det = DriverDrowsinessDetector(static_image_mode=True)
        det.fallback_mode = True
        det.face_mesh = None
        r_blank = det.process_frame(blank_frame)
        assert r_blank["face_detected"] is False
        assert r_blank["driver_status"] == "SEARCHING"

        r_noise = det.process_frame(noise_frame)
        assert r_noise["face_detected"] is False
        assert r_noise["driver_status"] == "SEARCHING"
        det.close()
