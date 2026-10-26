"""
Core Driver Drowsiness, Fatigue & Microsleep Detection Engine (The Detective)
Project: Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System
System Codename: DrowsyGuard DMS & Automotive Telemetry Engine
"""

import time
import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
import mediapipe as mp


class DriverDrowsinessDetector:
    """
    Production-Grade Facial Landmark & Driver Fatigue Perception Engine.
    Employs MediaPipe 3D FaceMesh to track 468 facial landmark coordinates,
    computes real-time Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
    and 3D Head Pose (Pitch/Yaw/Roll), and enforces smart stopwatch timer logic
    to distinguish natural blinks from dangerous microsleeps.
    """

    # Canonical 3D Human Face Model Anchors for Head Pose Estimation (PnP)
    # Coordinate system: Origin at nose tip.
    # +X points right (viewer's right, landmark 263).
    # +Y points down (chin is lower than nose in image, landmark 152).
    # +Z points into the face / away from camera (eyes/chin are deeper than nose tip).
    FACE_3D_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
        (0.0, 330.0, 65.0),          # Chin (Landmark 152)
        (-225.0, -170.0, 135.0),     # Left eye outer corner (Landmark 33)
        (225.0, -170.0, 135.0),      # Right eye outer corner (Landmark 263)
        (-150.0, 150.0, 125.0),      # Left mouth corner (Landmark 61)
        (150.0, 150.0, 125.0)        # Right mouth corner (Landmark 291)
    ], dtype=np.float64)

    # MediaPipe FaceMesh Landmark Indices for EAR, MAR, and Pose
    # Right eye landmarks: [outer, top1, top2, inner, bot2, bot1]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Left eye landmarks: [inner, top1, top2, outer, bot2, bot1]
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # Mouth landmarks: corners [61, 291], top [39, 0, 269], bottom [181, 17, 405]
    MOUTH_CORNER_INDICES = [61, 291]
    MOUTH_TOP_INDICES = [39, 0, 269]
    MOUTH_BOTTOM_INDICES = [181, 17, 405]

    # 6 Anchor indices corresponding to FACE_3D_MODEL_POINTS
    POSE_ANCHOR_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(
        self,
        ear_threshold: float = 0.22,
        consec_frames: int = 45,
        mar_threshold: float = 0.65,
        pitch_threshold: float = 25.0,
        static_image_mode: bool = False
    ):
        """
        Initializes the Driver Drowsiness Detector.

        Args:
            ear_threshold: Eye Aspect Ratio threshold (eyes closed if avg_ear < threshold).
            consec_frames: Number of consecutive frames eyes must remain closed to trigger microsleep siren (~1.5s at 30 FPS).
            mar_threshold: Mouth Aspect Ratio threshold (yawning if mar > threshold).
            pitch_threshold: Head downward pitch threshold in degrees (nodding off if pitch < -threshold).
            static_image_mode: Set True for single still photos; False for continuous video/webcam.
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.mar_threshold = mar_threshold
        self.pitch_threshold = pitch_threshold
        self.static_image_mode = static_image_mode

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
            static_image_mode=static_image_mode
        )

        # Internal State Machine (Smart Stopwatch)
        self.consecutive_closed_frames: int = 0
        self.consecutive_yawn_frames: int = 0
        self.total_blinks: int = 0
        self.prev_eye_state: str = "OPEN"
        self.frame_counter: int = 0

    def reset(self) -> None:
        """Resets temporal frame counters and state buffers for clean single-test evaluation."""
        self.consecutive_closed_frames = 0
        self.consecutive_yawn_frames = 0
        self.total_blinks = 0
        self.prev_eye_state = "OPEN"
        self.frame_counter = 0

    @staticmethod
    def calculate_ear(eye_pts: np.ndarray) -> float:
        """
        Computes the Eye Aspect Ratio (EAR) across 6 eye landmark coordinates:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Args:
            eye_pts: Array of shape (6, 2) representing (x, y) coordinates.

        Returns:
            float: Eye Aspect Ratio (typically 0.10 for closed eyes, 0.30+ for open eyes).
        """
        if len(eye_pts) != 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_pts

        # Vertical eyelid distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        # Horizontal eye corner width
        h = np.linalg.norm(p1 - p4)

        if h < 1e-6:
            return 0.0

        ear = float((v1 + v2) / (2.0 * h))
        return max(0.0, min(0.60, ear))

    @staticmethod
    def calculate_mar(mouth_corners: np.ndarray, mouth_top: np.ndarray, mouth_bottom: np.ndarray) -> float:
        """
        Computes the Mouth Aspect Ratio (MAR) to measure vertical lip opening relative to mouth width:
        MAR = (||t1 - b1|| + ||t2 - b2|| + ||t3 - b3||) / (2 * ||c1 - c2||)

        Returns:
            float: Mouth Aspect Ratio (typically 0.20 for closed mouth, 0.65+ for wide yawn).
        """
        if len(mouth_corners) != 2 or len(mouth_top) != 3 or len(mouth_bottom) != 3:
            return 0.0

        c1, c2 = mouth_corners
        h_width = np.linalg.norm(c1 - c2)
        if h_width < 1e-6:
            return 0.0

        v_dist = sum(np.linalg.norm(t - b) for t, b in zip(mouth_top, mouth_bottom))
        mar = float(v_dist / (2.0 * h_width))
        return max(0.0, min(1.50, mar))

    @staticmethod
    def estimate_head_pose(landmarks_px: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Estimates 3D Head Orientation (Pitch, Yaw, Roll) using Perspective-n-Point (cv2.solvePnP).

        Args:
            landmarks_px: Full pixel landmarks array of shape (468+, 2).
            frame_shape: Frame dimensions (H, W).

        Returns:
            Tuple of (pitch_deg, yaw_deg, roll_deg).
            pitch_deg: Negative = Slumping downward (nodding off); Positive = Looking upward.
        """
        h, w = frame_shape[:2]
        if len(landmarks_px) < max(DriverDrowsinessDetector.POSE_ANCHOR_INDICES) + 1:
            return 0.0, 0.0, 0.0

        # Extract 2D image points for pose anchors
        image_points = np.array([
            landmarks_px[idx] for idx in DriverDrowsinessDetector.POSE_ANCHOR_INDICES
        ], dtype=np.float64)

        # Approximate camera focal length and optical center
        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64) # Assuming minimal lens distortion

        success, rvec, tvec = cv2.solvePnP(
            DriverDrowsinessDetector.FACE_3D_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to 3x3 rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        # pitch: rotation around X axis (nodding)
        # yaw: rotation around Y axis (turning head left/right)
        # roll: rotation around Z axis (tilting head sideways)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))

        return pitch_deg, yaw_deg, roll_deg

    def process_frame(
        self,
        frame: np.ndarray,
        ear_threshold: Optional[float] = None,
        consec_frames: Optional[int] = None,
        mar_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end driver monitoring perception pipeline on an RGB frame:
        1. 3D Facial Landmark Mesh extraction (468 points).
        2. Eye Aspect Ratio (EAR) computation on left and right eyes.
        3. Mouth Aspect Ratio (MAR) computation for yawn detection.
        4. Head Pose estimation for nod-off and forward slump detection.
        5. Smart Stopwatch timer logic distinguishing normal blinks from critical microsleeps.

        Args:
            frame: RGB video frame as uint8 NumPy array of shape (H, W, 3).
            ear_threshold: Optional override for eye closure sensitivity.
            consec_frames: Optional override for alarm trigger delay.
            mar_threshold: Optional override for yawn sensitivity.

        Returns:
            Dict[str, Any]: Strongly typed telemetry dictionary conforming to architecture contract.
        """
        start_time = time.perf_counter()
        self.frame_counter += 1

        active_ear_thresh = ear_threshold if ear_threshold is not None else self.ear_threshold
        active_consec_limit = consec_frames if consec_frames is not None else self.consec_frames
        active_mar_thresh = mar_threshold if mar_threshold is not None else self.mar_threshold

        h, w = frame.shape[:2]

        # MediaPipe expects RGB format
        results = self.face_mesh.process(frame)

        # Baseline default telemetry when no face is detected
        if not results.multi_face_landmarks:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "frame_id": self.frame_counter,
                "timestamp_s": round(self.frame_counter / 30.0, 3),
                "face_detected": False,
                "driver_status": "SEARCHING",
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "mar": 0.0,
                "head_pitch_deg": 0.0,
                "head_yaw_deg": 0.0,
                "head_roll_deg": 0.0,
                "consecutive_closed_frames": 0,
                "consecutive_yawn_frames": 0,
                "total_blinks": self.total_blinks,
                "microsleep_alert": False,
                "yawn_alert": False,
                "head_slump_alert": False,
                "inference_latency_ms": round(latency_ms, 2),
                "landmarks_px": np.array([])
            }

        # Extract 2D pixel coordinates for all 468 landmarks
        face_lms = results.multi_face_landmarks[0]
        landmarks_px = np.array([
            [int(lm.x * w), int(lm.y * h)] for lm in face_lms.landmark
        ], dtype=np.int32)

        # ---------------------------------------------------------------------
        # 1. Compute Eye Aspect Ratio (EAR)
        # ---------------------------------------------------------------------
        right_eye_pts = np.array([landmarks_px[i] for i in self.RIGHT_EYE_INDICES], dtype=np.float64)
        left_eye_pts = np.array([landmarks_px[i] for i in self.LEFT_EYE_INDICES], dtype=np.float64)

        right_ear = self.calculate_ear(right_eye_pts)
        left_ear = self.calculate_ear(left_eye_pts)
        avg_ear = (right_ear + left_ear) / 2.0

        # ---------------------------------------------------------------------
        # 2. Compute Mouth Aspect Ratio (MAR)
        # ---------------------------------------------------------------------
        mouth_corners = np.array([landmarks_px[i] for i in self.MOUTH_CORNER_INDICES], dtype=np.float64)
        mouth_top = np.array([landmarks_px[i] for i in self.MOUTH_TOP_INDICES], dtype=np.float64)
        mouth_bottom = np.array([landmarks_px[i] for i in self.MOUTH_BOTTOM_INDICES], dtype=np.float64)

        mar = self.calculate_mar(mouth_corners, mouth_top, mouth_bottom)

        # ---------------------------------------------------------------------
        # 3. Compute Head Pose (Pitch, Yaw, Roll)
        # ---------------------------------------------------------------------
        pitch_deg, yaw_deg, roll_deg = self.estimate_head_pose(landmarks_px, (h, w))

        # ---------------------------------------------------------------------
        # 4. Smart Stopwatch: Natural Blink vs. Microsleep & Yawn Logic
        # ---------------------------------------------------------------------
        microsleep_alert = False
        yawn_alert = False
        head_slump_alert = False

        # Eye Closure Evaluation
        if avg_ear < active_ear_thresh:
            self.consecutive_closed_frames += 1
            if self.consecutive_closed_frames >= active_consec_limit:
                microsleep_alert = True
        else:
            # Check if this transition completed a natural blink (eyes closed for 2 to 15 frames)
            if 2 <= self.consecutive_closed_frames < active_consec_limit:
                self.total_blinks += 1
            self.consecutive_closed_frames = 0

        # Yawn Evaluation (sustained wide mouth opening >= 12 frames / ~0.4s, or 1 frame in static image mode)
        yawn_limit = 1 if active_consec_limit == 1 else 12
        if mar >= active_mar_thresh:
            self.consecutive_yawn_frames += 1
            if self.consecutive_yawn_frames >= yawn_limit:
                yawn_alert = True
        else:
            self.consecutive_yawn_frames = 0

        # Head Slump / Nod-off Evaluation (pitch tilted downward past threshold)
        if pitch_deg < -self.pitch_threshold:
            head_slump_alert = True
            # If head is slumped down while eyes are closed even briefly, escalate alert
            if self.consecutive_closed_frames >= 15:
                microsleep_alert = True

        # ---------------------------------------------------------------------
        # 5. Determine Overall Operational Driver Status
        # ---------------------------------------------------------------------
        if microsleep_alert:
            driver_status = "CRITICAL"
        elif yawn_alert or head_slump_alert:
            driver_status = "DROWSY"
        else:
            driver_status = "ATTENTIVE"

        # Select highlighted subset of eye & mouth landmarks for Painter HUD
        annotated_subset_indices = self.RIGHT_EYE_INDICES + self.LEFT_EYE_INDICES + self.MOUTH_CORNER_INDICES + self.MOUTH_TOP_INDICES + self.MOUTH_BOTTOM_INDICES
        hud_landmarks_px = np.array([landmarks_px[i] for i in annotated_subset_indices], dtype=np.int32)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "frame_id": self.frame_counter,
            "timestamp_s": round(self.frame_counter / 30.0, 3),
            "face_detected": True,
            "driver_status": driver_status,
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "avg_ear": round(avg_ear, 3),
            "mar": round(mar, 3),
            "head_pitch_deg": round(pitch_deg, 1),
            "head_yaw_deg": round(yaw_deg, 1),
            "head_roll_deg": round(roll_deg, 1),
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "consecutive_yawn_frames": self.consecutive_yawn_frames,
            "total_blinks": self.total_blinks,
            "microsleep_alert": microsleep_alert,
            "yawn_alert": yawn_alert,
            "head_slump_alert": head_slump_alert,
            "inference_latency_ms": round(latency_ms, 2),
            "landmarks_px": hud_landmarks_px,
            "all_landmarks_px": landmarks_px
        }

    def close(self) -> None:
        """Closes the underlying MediaPipe FaceMesh engine and releases resources."""
        if hasattr(self, "face_mesh") and self.face_mesh is not None:
            self.face_mesh.close()
