# Engineering Task Queue & Implementation Blueprint

**Project Title:** Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System  
**System Codename:** `DrowsyGuard DMS & Automotive Telemetry Engine`  
**Document Version:** 1.0.0  
**Phase:** 3 — Tech Lead Agent  
**Framework:** `MASTER_SDLC_AGENTS`  
**Status:** Under Review (Gate 3 Pending Approval)  
**Date:** 2026-10-24  

---

## 1. Architectural-to-Physical File Mapping

In Phase 2, the Technical Architect defined the logical subsystems and data dictionaries. The Tech Lead now maps those logical subsystems to physical modular files in the codebase:

| Logical Subsystem (Phase 2) | Target Physical File | Lead Agent Assigned |
| :--- | :--- | :---: |
| **Foundation & Dependencies** | `requirements.txt`, `packages.txt` | Software Engineer (Phase 4) |
| **Media Ingestion & The Painter** | `src/utils.py` | Software Engineer (Phase 4) |
| **The Detective (Perception Engine)**| `src/detector.py` | Software Engineer (Phase 4) |
| **Driver Console Presentation** | `src/app.py` | Software Engineer (Phase 4) |
| **QA Test Suite & Verification** | `tests/conftest.py`, `tests/test_*.py` | QA Test Engineer (Phase 5) |
| **Defensive Security Audit** | `reports/security-audit.md` | Security Auditor (Phase 6) |
| **Production DevOps Deliverables** | `Dockerfile`, `docker-compose.yml`, `README.md`, `plans/detailed-design.*` | DevOps Engineer (Phase 7) |

---

## 2. Sequential Task Queue & Definitions of Done (DoD)

### Task 1: Environment & Dependency Configuration
* **Agent Role:** Software Engineer Agent
* **Target Files:** `requirements.txt`, `packages.txt`
* **Technical Scope:**
  * Define pinned dependencies: `mediapipe>=0.10.0`, `opencv-python-headless>=4.9.0`, `numpy<2.0.0`, `Pillow>=10.0.0`, `pandas>=2.0.0`, `streamlit>=1.35.0`, `pytest>=8.0.0`, `torch>=2.2.0`, `torchvision>=0.17.0`.
  * Specify Debian OS packages in `packages.txt` for Streamlit Cloud and headless Docker: `libgl1`, `libglib2.0-0`.
* **Definition of Done (DoD):**
  1. Clean installation in fresh virtual environment without dependency conflicts.
  2. Zero ABI incompatibility between NumPy and OpenCV/MediaPipe.

---

### Task 2: Utility, Image Processing & Painter HUD Module
* **Agent Role:** Software Engineer Agent
* **Target File:** `src/utils.py`
* **Technical Scope:**
  * `validate_media_input(file_bytes, filename)`: Enforce strict file extension and header checks (`.mp4`, `.avi`, `.mov`, `.jpg`, `.png`).
  * `load_and_validate_image(input_data)`: Normalize bytes, PIL images, and NumPy arrays into 3-channel RGB uint8 format.
  * `extract_video_frame_iterator(source, max_frames)`: Generator yielding sequential RGB frames with bounded loop limits to prevent memory leaks.
  * `draw_drowsiness_overlay(frame, landmarks, telemetry)`:
    * Render cyan anti-aliased contours on upper/lower eyelids.
    * Render amber contours tracing outer/inner lips.
    * Draw dynamic driver bounding box: Emerald Green (Attentive), Amber (Drowsy / Yawn), Crimson Red (Critical Microsleep).
    * Composite HUD status banner with plain-English badges.
  * `generate_audio_alert_b64()`: Generates synthesized in-browser audio chime/siren wave buffers for emergency wake-up calls.
  * `format_telemetry_csv(records)`: Serializes per-frame metrics into standard RFC 4180 CSV logs with formula injection protection.
  * Defense-in-depth: Cap `Image.MAX_IMAGE_PIXELS = 50_000_000` to prevent decompression bomb DoS.
* **Definition of Done (DoD):**
  1. All functions are pure, modular, and fully type-hinted with docstrings.
  2. Successfully verified with unit tests for empty, corrupted, and valid inputs.

---

### Task 3: Facial Landmark & Fatigue Perception Engine (The Detective)
* **Agent Role:** Software Engineer Agent
* **Target File:** `src/detector.py`
* **Technical Scope:**
  * Implement `DriverDrowsinessDetector` class:
    * `__init__(ear_threshold=0.22, consec_frames=45, mar_threshold=0.65, pitch_threshold=25.0)`: Initialize MediaPipe FaceMesh engine.
    * `extract_landmarks(frame)`: Detect face and extract 468 normalized 3D facial landmark coordinates.
    * `calculate_ear(eye_landmarks)`: Compute Eye Aspect Ratio:
      $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
    * `calculate_mar(mouth_landmarks)`: Compute Mouth Aspect Ratio:
      $$\text{MAR} = \frac{\|m_2 - m_8\| + \|m_3 - m_7\| + \|m_4 - m_6\|}{2 \|m_1 - m_5\|}$$
    * `estimate_head_pose(landmarks, frame_shape)`: Compute 3D rotation vectors and extract head pitch, yaw, and roll angles in degrees.
    * `process_frame(frame, ...)`: Execute end-to-end perception pipeline, manage consecutive closed-frame counters (The Smart Stopwatch), evaluate state transitions (`ATTENTIVE` $\rightarrow$ `DROWSY` $\rightarrow$ `CRITICAL`), and return strongly typed telemetry dictionary.
    * `reset()`: Clear internal frame counters and historical states for clean single-frame/video resets.
* **Definition of Done (DoD):**
  1. Output matches Phase 2 telemetry schema contract 100%.
  2. Latency benchmark executes in $< 35\text{ms}$ on standard CPU.
  3. Accurately flags natural blinks as non-alarms and microsleeps ($\ge 1.5$s) as critical sirens.

---

### Task 4: Interactive Web Dashboard & Driver Console
* **Agent Role:** Software Engineer Agent
* **Target File:** `src/app.py`
* **Technical Scope:**
  * Streamlit web application with modern dark-slate automotive styling.
  * Dual Ingestion Modes:
    * Mode A: **Live Interactive Webcam Feed** (`st.camera_input` or live streaming frame processor).
    * Mode B: **Preset Driving Demos** (Alert Driver, Yawning Driver, Microsleep Driver) with real video clips and benchmark photos.
  * Four Human-Friendly Metric Cards:
    * `1. Driver Alertness`: `🟢 ATTENTIVE (Driver Alert)` vs `🟡 DROWSY (Yawning)` vs `🚨 CRITICAL (Microsleep Alarm!)`.
    * `2. Eye Opening Level`: `👁️ Eyes Open (0.32)` vs `⚠️ Eyes Closed (0.14)`.
    * `3. Yawn / Mouth State`: `👄 Mouth Closed (0.35)` vs `🥱 Yawn Detected (0.72)`.
    * `4. Head Position`: `👤 Facing Forward (0°)` vs `⚠️ Nodding Off (Slumped Down)`.
  * Sidebar parameter sliders with bracketed plain-English guidance from PRD Section 7.
  * Audio siren alert integration playing in-browser audio when `microsleep_alert == True`.
  * 1-Click CSV telemetry log download button.
  * Master framework attribution badge: `🤖 MASTER_SDLC_AGENTS: 7-Agent Master Framework`.
* **Definition of Done (DoD):**
  1. Dashboard launches smoothly via `streamlit run src/app.py`.
  2. Seamless real-time switching between live camera and preset demos without session crashes.

---

### Task 5: Automated Testing & Verification Suite
* **Agent Role:** QA / Test Engineer Agent
* **Target Files:** `tests/conftest.py`, `tests/test_utils.py`, `tests/test_detector.py`, `tests/generate_test_results.py`
* **Technical Scope:**
  * `test_utils.py`: Unit test media validation, corrupted uploads, image conversions, HUD graphics rendering, and CSV exports.
  * `test_detector.py`: Integration test facial landmark extraction, EAR math on open vs closed eyes, MAR math on yawning mouth, head pitch angles, Smart Stopwatch consecutive frame accumulation, and reset lifecycle.
  * `generate_test_results.py`: Batch runner generating visual annotated output images saved in `tests/test_results/` for human verification.
* **Definition of Done (DoD):**
  1. 100% automated test pass rate with PyTest (`pytest -v tests`).
  2. Zero source code modifications in `src/` during Phase 5 (code freeze guardrail).

---

### Task 6: Defensive Security & Vulnerability Audit
* **Agent Role:** Defensive Security Auditor Agent
* **Target File:** `reports/security-audit.md`
* **Technical Scope:**
  * SAST static analysis for code execution vulnerabilities (`eval`, `exec`, `os.system`, `subprocess`).
  * In-memory privacy verification: Ensure zero webcam frames are stored to disk or transmitted externally.
  * Decompression bomb protection audit (`Image.MAX_IMAGE_PIXELS`).
  * Temporary file lifecycle audit (`tempfile.NamedTemporaryFile` with `finally: os.remove`).
  * Dependency supply chain vulnerability check.
* **Definition of Done (DoD):**
  1. Formal report authored with explicit verdict: `SAFE TO COMMIT`.

---

### Task 7: Production Containerization & Comprehensive Documentation
* **Agent Role:** DevOps & Release Engineer Agent
* **Target Files:**
  * `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`
  * `README.md`
  * `plans/detailed-design.md`, `plans/detailed-design.docx`, `plans/detailed-design.pdf`
  * `Deployment_and_Customer_Handover_Guide.docx` & `.pdf`
  * `Docker_Generation_and_Testing_Manual.docx` & `.pdf`
* **Technical Scope:**
  * Multi-architecture Docker configuration with non-root `appuser` and automated healthcheck probes.
  * Comprehensive `README.md` featuring `.gemini` framework rules and technology badges.
  * Detailed Design Document containing the complete 4-step detection logic, EAR/MAR mathematical formulations, comparative backbones guide, and edge-case resolutions.
  * Git repository release commit and push assistance.
* **Definition of Done (DoD):**
  1. `docker compose config` validates with zero errors.
  2. All documents generated in both `.docx` and `.pdf` formats.

---

## 3. Strict Execution Protocol

In strict adherence to the **`MASTER_SDLC_AGENTS`** framework:
1. **One Task at a Time:** Tasks 1 through 7 will be executed strictly sequentially.
2. **Explicit User Approval:** No code will be written in `src/` until the user formally approves **Gate 3**.
