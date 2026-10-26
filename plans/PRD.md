# Product Requirements Document (PRD)

**Project Title:** Real-Time AI Driver Drowsiness, Fatigue & Microsleep Detection System  
**System Codename:** `DrowsyGuard DMS & Automotive Telemetry Engine`  
**Document Version:** 1.0.0  
**Phase:** 1 — Product Manager Agent  
**Framework:** `MASTER_SDLC_AGENTS`  
**Status:** Under Review (Gate 1 Pending Approval)  
**Date:** 2026-10-24  

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
According to the National Highway Traffic Safety Administration (NHTSA) and the World Health Organization (WHO), drowsy driving is responsible for over **100,000 police-reported crashes and thousands of fatalities annually**. Driver fatigue causes degraded reaction time, involuntary "microsleeps" (brief lapses of consciousness lasting 1 to 5 seconds), and loss of vehicle control. 

Commercial fleet operators, long-haul trucking companies, and commuter passenger vehicles urgently require an automated, real-time **Driver Monitoring System (DMS)** that continuously analyzes driver alertness without requiring intrusive wearable sensors.

### 1.2 Product Vision
`DrowsyGuard` is a non-intrusive, camera-based computer vision Driver Monitoring System (DMS). Operating on commodity hardware (standard x86 CPU and standard webcams/dashcams), it tracks 3D facial landmarks, computes real-time physiological indicators (Eye Aspect Ratio, Mouth Aspect Ratio, Head Pose), and triggers deterministic visual and audio alerts when early fatigue or critical microsleep occurs.

### 1.3 Core Perception Logic: The 4-Step Drowsiness Detection Pipeline

To ensure total transparency for both engineering teams and non-technical stakeholders, the perception system operates through the following 4-step pipeline:

```
[In-Cabin Camera / Dashcam / Webcam]
               │
               ▼
   Step 1: Locates the Driver's Face in the Cabin
   (Isolates the driver's head, ignoring seats, steering wheel, and windows)
               │
               ▼
   Step 2: Maps 3D Facial Landmark Points (The "Digital Mask")
   (Locks 6 coordinates per eye, 8 coordinates per mouth, and nose bridge)
               │
               ▼
   Step 3: Measures Eyelid Separation (Eye Aspect Ratio - EAR)
   (Calculates EAR = Vertical Eyelid Distance / Eye Corner Width)
               │
               ▼
   Step 4: Smart Stopwatch Timer Logic (Blink vs. Microsleep)
   (Natural Blink: < 0.25s -> Ignored; Microsleep: >= 1.5s -> Trigger Siren)
```

1. **Step 1 — Locating the Driver's Face in the Cabin ("Find the Person"):**  
   The camera frame contains steering wheels, headrests, and background scenery. The deep learning face detector scans the frame 30 times a second and isolates the driver's head bounding box.
2. **Step 2 — Mapping 3D Facial Landmark Points ("The Digital Mask"):**  
   The model anchors precise mathematical coordinates around the facial anatomy:
   * 6 points around each eye ($p_1, p_4$ at corners; $p_2, p_3$ on upper eyelid; $p_5, p_6$ on lower eyelid).
   * 8 points tracing upper and lower lips.
   * Coordinate anchors on the nose and jawline for 3D head pitch/yaw/roll.
3. **Step 3 — Measuring Eyelid Separation via Eye Aspect Ratio ("The Measuring Ruler"):**  
   * **Eyes Wide Open:** Vertical eyelid opening is large ($\text{EAR} \approx 0.30\text{--}0.35$) $\rightarrow$ Status: **🟢 ATTENTIVE**.
   * **Eyes Closed:** Top eyelid contacts bottom eyelid; vertical distance drops near zero ($\text{EAR} < 0.20$).
4. **Step 4 — Smart Stopwatch Timer Logic ("Blink vs. Sleep"):**  
   * **Normal Blink ($< 0.25\text{s}$):** Eyes close and reopen within 5–8 frames. The system logs a normal blink and resets the timer to zero without sounding an alarm.
   * **Microsleep ($\ge 1.5\text{s}$):** Eyelids remain closed continuously for $\ge 45$ consecutive frames. The system detects a dangerous loss of consciousness and triggers the **🚨 CRITICAL ALARM & SIREN**.

---

## 2. User Personas & Use Cases

### 2.1 Target User Personas
* **Persona 1: Commercial Fleet Safety Officer (Enterprise):** Needs an automated telemetry audit trail (CSV export) of driver drowsiness incidents, blink rates, and fatigue occurrences across highway shifts.
* **Persona 2: Passenger Vehicle Operator (Consumer):** Needs an intuitive, non-technical dashboard that operates with zero calibration and gives clear, non-intrusive alerts before catastrophic micro-sleep occurs.
* **Persona 3: Autonomous Vehicle Safety Engineer (Developer):** Requires reliable driver handover readiness metrics to verify driver alertness before disengaging Level 2/3 autonomous autopilot.

### 2.2 Primary Use Cases
* **UC-1 (Live Driver Monitoring):** Driver activates their laptop/cabin webcam; the system immediately locks onto facial landmarks, starts tracking blink frequency, and sounds a progressive chime if eyes remain closed.
* **UC-2 (Preset Fatigue Simulation & Verification):** Safety auditor selects preset video clips (Alert Driver, Yawning Driver, Prolonged Microsleep Driver) to verify algorithm accuracy and alarm thresholds without needing a live camera.
* **UC-3 (Fleet Compliance Export):** Safety manager downloads the complete frame-by-frame telemetry log (RFC 4180 CSV) to inspect time-stamped Eye Aspect Ratio (EAR) drops and alarm trigger events.

---

## 3. Functional Requirements (FRs)

* **FR-1 (3D Facial Landmark Tracking):** The system shall detect driver faces in real time and track 3D facial landmark coordinates across the eye contours, eyelids, outer/inner lips, and nose bridge.
* **FR-2 (Eye Aspect Ratio & Microsleep Detection):** The system shall compute the Eye Aspect Ratio (EAR) for both left and right eyes on every frame. If the average EAR drops below the defined threshold for longer than the alert delay (default: $\ge 1.5$ seconds / 45 frames), the system shall trigger an immediate **CRITICAL MICROSLEEP ALARM**.
* **FR-3 (Mouth Aspect Ratio & Yawn Detection):** The system shall compute the Mouth Aspect Ratio (MAR) to measure vertical lip opening relative to mouth width. Sustained openings exceeding the threshold shall trigger a **DROWSY / YAWN WARNING**.
* **FR-4 (Head Pose & Slump / Nodding Tracking):** The system shall estimate 3D head pitch (nodding down), yaw (looking away), and roll (tilting). Downward head slumping past the safe angle shall escalate alertness warnings.
* **FR-5 (Multi-Tier Alert Escalation System):**
  * **Level 1: Safe / Attentive (Green):** Eyes open ($\text{EAR} \ge 0.25$), normal blink frequency, head forward.
  * **Level 2: Drowsy / Warning (Amber):** Driver is yawning ($\text{MAR} \ge 0.60$) or exhibiting erratic blink patterns.
  * **Level 3: Critical Alarm / Microsleep (Crimson Red):** Driver's eyes closed for $\ge 1.5$s or head slumped down. Triggers prominent flashing on-screen warning and audible buzzer cue.
* **FR-6 (Dual Ingestion Support):**
  * **Mode A: Live Camera Feed:** Real-time webcam ingestion directly from user's local camera.
  * **Mode B: Preset Benchmark Demos:** Curated driving video clips and still photos for immediate verification (Active Alert Driver, Yawning Driver, Microsleep Driver).
* **FR-7 (1-Click Telemetry Audit Log):** The system shall export full frame-by-frame RFC 4180 CSV logs containing timestamp, EAR, MAR, head pitch angle, blink counter, and alert status for fleet compliance analysis.

---

## 4. Non-Functional Requirements (NFRs)

* **NFR-1 (Real-Time Performance on CPU):** The inference and analytical pipeline shall execute in under **40ms per frame** ($\ge 25\text{--}30\text{ FPS}$) on standard commodity x86 CPUs without requiring GPU acceleration.
* **NFR-2 (Memory Footprint & Leak Prevention):** Continuous multi-hour video and webcam processing shall consume under **500 MB RAM** with zero memory leaks across extended monitoring.
* **NFR-3 (Privacy & Security First):** All webcam video frames must be processed transiently in memory. No driver camera images or video recordings shall be permanently stored on disk or transmitted across external networks.
* **NFR-4 (Turnkey Containerization):** The entire application shall be packageable into a production Docker container with automated healthcheck probes runnable via a single `docker compose up` command.
* **NFR-5 (Deterministic Testing & Coverage):** The system shall achieve 100% automated test pass rate across unit and integration suites prior to release.

---

## 5. Visual Standards & HUD Telemetry Contract

| Visual Element | Specification | Visual Representation |
| :--- | :--- | :--- |
| **Eye Landmark Contours** | Anti-aliased facial lines | **Cyan / Electric Blue** `RGB(0, 220, 255)` |
| **Mouth Landmark Contours**| Anti-aliased lip tracking line | **Bright Amber** `RGB(255, 180, 0)` |
| **Facial Bounding Box** | Thin rounded tracker border | **Emerald Green** (Alert) \| **Crimson Red** (Sleeping) |
| **Attentive Status Badge** | Eyes open, attentive | **Green Badge:** `🟢 ATTENTIVE (Driver Alert)` |
| **Fatigue / Yawn Badge** | Mouth wide open / Yawn | **Amber Badge:** `🟡 DROWSY (Yawning Detected)` |
| **Microsleep Siren Badge** | Eyes closed $\ge 1.5\text{s}$ | **Crimson Red Badge:** `🚨 CRITICAL (MICROSLEEP ALARM!)` |

---

## 6. Definition of Done (DoD) for Gate 1 Approval

This Product Requirements Document is complete when:
1. Product scope, problem definition, and user use cases are documented.
2. The computer vision perception approach (EAR, MAR, Head Pose) is formalized.
3. Multi-tier alert escalation levels and NFR latency budgets are defined.
4. Plain-English end-user parameter definitions and bracketed tooltips are specified in Section 7.
5. The user provides explicit Human-in-the-Loop (HITL) sign-off to proceed to Phase 2 (Technical Architect).

---

## 7. End-User Plain-English Terminology & Parameter Specifications

To ensure the final product is immediately understandable by non-technical vehicle drivers and safety auditors without requiring calculus or computer vision degrees, all metrics and parameters follow these user-facing definitions:

### 7.1 Dashboard Metric Cards (Driver-Facing Display)

| Metric Card | Technical Variable | Plain-English Label | End-User Interpretation |
| :--- | :--- | :--- | :--- |
| **1. Driver Alertness** | `driver_status` | **🟢 ATTENTIVE (Driver Alert)** vs **🟡 DROWSY (Fatigue)** vs **🚨 CRITICAL (Microsleep Alarm!)** | Confirms whether the driver is awake, exhibiting early fatigue signs, or falling asleep at the wheel. |
| **2. Eye Opening Level** | `eye_aspect_ratio` | **👁️ Eyes Open (e.g. 0.32)** vs **⚠️ Eyes Closed (e.g. 0.14)** | Shows how wide the driver's eyes are open. Values $> 0.25$ mean eyes are open; values $< 0.20$ indicate closed eyelids. |
| **3. Yawn / Mouth State** | `mouth_aspect_ratio`| **👄 Mouth Closed (e.g. 0.35)** vs **🥱 Yawn Detected (e.g. 0.72)** | Measures vertical mouth opening. Automatically flags fatigue yawning when the mouth stays wide open. |
| **4. Head Position** | `head_pitch_deg` | **👤 Facing Forward (0°)** vs **⚠️ Nodding Off (Slumped Down)** | Tracks whether the driver is looking at the road ahead or if their head is slumping forward from exhaustion. |

### 7.2 Driver Assistance Control Settings (Sidebar Tuning with Bracketed Guidance)

| UI Label Displayed to End User | Technical Name | Default | When to Adjust? |
| :--- | :--- | :---: | :--- |
| **🎯 Eye Closure Sensitivity (Lower = Deep Sleep Only \| Higher = Catch Drowsy Squints)** | `ear_threshold` | `0.22` | Slide **LEFT (0.18)** if driver naturally has narrower eyes.<br>Slide **RIGHT (0.26)** for strict detection of heavy, half-closed eyelids. |
| **⏱️ Sleep Alarm Trigger Delay (Short: 1.0s \| Normal: 1.5s \| Relaxed: 2.5s)** | `consec_frames` | `1.5s (45 frames)` | Controls how long eyes must stay closed before siren sounds.<br>**1.0s (Strict):** Highway driving.<br>**1.5s (Standard):** Normal driving.<br>**2.5s (Relaxed):** Avoids alarms on slow blinks. |
| **🥱 Yawn Sensitivity (Lower = Deep Yawns Only \| Higher = Detect Small Yawns)** | `mar_threshold` | `0.65` | Slide **LEFT (0.75)** to trigger only on full wide yawns.<br>Slide **RIGHT (0.55)** to catch early mouth fatigue. |
| **📐 Head Nodding Sensitivity (Strict: 15° Down \| Normal: 25° Down \| Relaxed: 35° Down)** | `pitch_threshold` | `25°` | Controls the downward tilt angle before sounding a nodding-off alarm. |
| **⏱️ Video Clip Length (Short 2s: 60 \| Standard 4s: 120 \| Long 10s: 300)** | `max_frames` | `120 frames` | Quick 2-second check (60 frames) vs Full trip review (120–300 frames). |
