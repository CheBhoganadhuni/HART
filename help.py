import sys

def print_manual():
    manual = """
================================================================================
                        MOT-ATTENDANCE SYSTEM MANUAL
================================================================================

1. PROJECT OVERVIEW
-------------------
This is a High-Performance, Modular Face Recognition Attendance System.
It uses modern AI (YOLOv11 + InsightFace) for real-time tracking and recognition.
Key Features:
- Geometric 3D Pose Registration (Anti-Spoofing & Quality Check).
- Multi-Vector Embedding Cache: system learns each student across sessions (persistent per-section cache).
- "Pulse Window" Heartbeat Logic (100% accurate presence time with interval tracking).
- Handling of "Unknown" faces (Visitor tracking with image snapshots).
- Session-based Architecture with SQL Persistence.
- AI Orchestrator Log: real-time commentary on every recognition decision via the web dashboard.

2. FILE STRUCTURE & CORE COMPONENTS
-----------------------------------
- main.py: The Heart of the system. Runs the attendance session.
- register_student.py: Geometric Registration tool. Enrolls new students.
- manage_users.py: Admin dashboard. Manage students, sections, and view logs.
- core/db_manager.py: Database logic. Usage: `sqlite3 data/attendance.db`.
- botsort_custom.yaml: Configuration for the YOLO tracker (Motion models).
- help.py: This manual file.

3. HOW TO USE (COMMANDS)
------------------------

A. REGISTER A NEW STUDENT
   Command: python3 register_student.py
   - Asks for: Section (e.g., CSE_A), Student ID (Unique), Name.
   - Controls: 'p' (Pause), 'r' (Restart), 'q' (Quit).
   - Validation: Checks if ID exists. Warns if multiple faces detected.

B. MANAGE SYSTEM
   Command: python3 manage_users.py
   - Options:
     1. List Students (Shows Section, ID, Name, Status).
     2. Delete Student (Safe delete from DB + Embeddings + Cache).
     3. View Sessions (Summary of past classes).
     4. Manage Sections (Create/Delete sections).

C. RUN ATTENDANCE SESSION
   Command: python3 main.py --section <SECTION_NAME> [OPTIONS]
   
   Required Argument:
   --section <NAME> : Loads specific class data (e.g., CSE_A). Modular loading.

   Optional Arguments:
   --session <NAME> : Name the session (Default: Session_<timestamp>).
   --source <PATH>  : Video file path or '0' for Webcam (Default: 0).
   --export         : Save session video to `data/exports/`.
   (Cache is now auto-managed: data/section_caches/<section>.pkl is loaded and updated automatically.)

   Example:
   python3 main.py --section CSE_A --export

4. SYSTEM CAPABILITIES & LOGIC
------------------------------
- Heartbeat: Updates status to 'Present' every 10s ONLY if seen in that window.
             If a student leaves, they are marked 'Absent' automatically.
- Labels:
  [Green] Name           : Verified Face.
  [Blue]  Name (Re-ID)   : Fast recognition via session cache (<6s).
  [Orange] Identifying...: AI is processing the face.
  [Gray]  Visitor ID:123 : Unknown face (Logged to DB).
- Unknowns: Saved to `data/unknown_faces/` and `unknown_detections` table.
- Session Closure: Safely closes DB connection on exit (Press 'q') ('p' (Pause)).

5. DATABASE STRUCTURE (data/attendance.db)
------------------------------------------
Tables:
1. students:
   - id, student_id (Unique), name, section, created_at.

2. sessions:
   - id, session_name, section, status (ACTIVE/CLOSED), start_time, end_time.

3. session_attendance:
   - id, session_id, session_name, student_id, status (Present/Absent), last_seen.

4. attendance_intervals:
   - id, session_id, student_id, start_time, end_time.
   - One row per continuous presence window. Used for generating per-student time-in-room metrics.

5. unknown_detections:
   - id, session_id, session_name, track_id, image_path (Path to saved face).

6. DATA FOLDERS
---------------
- data/embeddings/         : Master face signatures per section (.pkl) — written at registration.
- data/section_caches/     : Per-section learning cache (.pkl) — auto-updated every session.
                             This is the "system learning" store. One file per section, e.g. CSE_A.pkl.
                             Built from live embeddings; speeds up recognition for returning students.
- data/unknown_faces/      : Snapshots of unrecognized individuals (jpg, one per detection).
- data/exports/            : Recorded session videos (when --export flag is used).
- data/slm_events.json     : Live JSON feed consumed by the AI Orchestrator Log in the web dashboard.
                             Written by main.py on every significant recognition event.

================================================================================
    For further assistance, view the source code in `main.py` or `core/`.
================================================================================
"""
    print(manual)

if __name__ == "__main__":
    print_manual()
