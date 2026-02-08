import os
import time
import json
import sqlite3
import subprocess
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Global Process Handle (Simple singleton for demo)
CV_PROCESS = None
SESSION_METADATA = {}
PROCESS_TYPE = None  # "webcam" | "mp4" | None - Global lock
MP4_PROGRESS = {"frame": 0, "total": 0, "status": "idle"}  # Progress tracking

# Paths
PROJECT_ROOT = settings.BASE_DIR.parent
COMMAND_PATH = os.path.join(PROJECT_ROOT, "data/commands.json")
STREAM_PATH = os.path.join(PROJECT_ROOT, "data/live_stream.jpg")
DB_PATH = os.path.join(PROJECT_ROOT, "data/attendance.db")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "data/embeddings")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data/persistent_cache")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data/uploads")
MP4_PROGRESS_PATH = os.path.join(PROJECT_ROOT, "data/mp4_progress.json")


def index(request):
    """Renders the main dashboard."""
    # Get available sections
    sections = []
    if os.path.exists(EMBEDDINGS_DIR):
        for f in os.listdir(EMBEDDINGS_DIR):
            if f.endswith(".pkl"):
                sections.append(f.replace(".pkl", ""))
    
    # Get available cache files sorted by modification time (newest first)
    caches = []
    if os.path.exists(CACHE_DIR):
        cache_files = []
        for f in os.listdir(CACHE_DIR):
            if f.startswith("cache_") and f.endswith(".pkl"):
                full_path = os.path.join(CACHE_DIR, f)
                mtime = os.path.getmtime(full_path)
                cache_files.append((f, mtime))
        # Sort by modification time descending (newest first)
        cache_files.sort(key=lambda x: x[1], reverse=True)
        caches = [f[0] for f in cache_files]
    
    # Check if any processing is running
    process_running = CV_PROCESS is not None and CV_PROCESS.poll() is None
    is_mp4_running = PROCESS_TYPE == "mp4" and process_running
    is_webcam_running = PROCESS_TYPE == "webcam" and process_running
    
    context = {
        "sections": sorted(sections),
        "caches": caches,  # All cache files, sorted by date
        "is_running": process_running,
        "is_busy": process_running,  # Block for ANY session (webcam OR MP4)
        "busy_type": "webcam" if is_webcam_running else ("MP4" if is_mp4_running else None),
        "hide_hamburger": process_running  # Hide hamburger if ANY session is active
    }
    return render(request, "dashboard/index.html", context)


def gen_frames():
    """Generator for MJPEG stream.
    
    Waits for live_stream.jpg to appear, then continuously streams frames.
    Streams every ~40ms (25 FPS) as long as file exists.
    """
    # Wait up to 15 seconds for first frame to appear
    for _ in range(375):  # 375 * 40ms = 15 seconds
        if os.path.exists(STREAM_PATH):
            break
        time.sleep(0.04)
    
    # Continuously stream frames while file exists
    missing_count = 0
    
    while True:
        try:
            if os.path.exists(STREAM_PATH):
                missing_count = 0
                with open(STREAM_PATH, "rb") as f:
                    frame_data = f.read()
                
                if frame_data and len(frame_data) > 100:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                missing_count += 1
                # Exit after 3 seconds of missing file (session likely stopped)
                if missing_count > 75:
                    break
        except (IOError, OSError):
            # File being written, just skip this iteration
            pass
        except GeneratorExit:
            # Browser disconnected
            break
        except Exception:
            pass
        
        time.sleep(0.04)  # ~25 FPS cap


def video_feed(request):
    """Returns the MJPEG stream."""
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def check_busy(request):
    """Returns current system busy state - for pre-start checks."""
    process_running = CV_PROCESS is not None and CV_PROCESS.poll() is None
    return JsonResponse({
        "busy": process_running,
        "type": PROCESS_TYPE if process_running else None
    })


@csrf_exempt
def start_session(request):
    """Starts the main.py subprocess for webcam."""
    global CV_PROCESS, SESSION_METADATA, PROCESS_TYPE
    
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    # Global lock check
    if CV_PROCESS is not None and CV_PROCESS.poll() is None:
        return JsonResponse({"error": f"System busy ({PROCESS_TYPE})", "status": "busy"}, status=409)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    section = data.get("section")
    session_name = data.get("session", "").strip()
    source = data.get("source", "0").strip()
    cache_file = data.get("cache", "")  # Optional cache preload
    export_enabled = data.get("export", True)
    
    if not section:
        return JsonResponse({"error": "Section is required"}, status=400)
    
    # Delete old live stream file to avoid showing stale frames
    try:
        if os.path.exists(STREAM_PATH):
            os.remove(STREAM_PATH)
    except Exception:
        pass
    
    # Construct Command
    cmd = [
        os.path.join(PROJECT_ROOT, "venv/bin/python3"), 
        os.path.join(PROJECT_ROOT, "main.py"),
        "--section", section,
        "--source", source
    ]
    
    if export_enabled:
        cmd.append("--export")
    
    if session_name:
        cmd.extend(["--session", session_name])
    
    if cache_file:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        if os.path.exists(cache_path):
            cmd.extend(["--cache", cache_path])
    
    try:
        CV_PROCESS = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        PROCESS_TYPE = "webcam"
        SESSION_METADATA = {"section": section, "start_time": time.time()}
        return JsonResponse({"status": "started", "pid": CV_PROCESS.pid})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def stop_session(request):
    """Sends stop signal to main.py."""
    global CV_PROCESS, PROCESS_TYPE
    
    # Write Stop Command
    try:
        with open(COMMAND_PATH, "w") as f:
            json.dump({"stop": True}, f)
    except Exception:
        pass
    
    # Early return if no process running
    if CV_PROCESS is None:
        return JsonResponse({"status": "stopped"})
    
    # Wait for process to exit gracefully
    for _ in range(10):  # 5 seconds total
        if CV_PROCESS.poll() is not None:
            break
        time.sleep(0.5)
    
    if CV_PROCESS.poll() is None:
        CV_PROCESS.terminate()
    
    CV_PROCESS = None
    PROCESS_TYPE = None
    return JsonResponse({"status": "stopped"})


@csrf_exempt
def control_session(request):
    """Pause/Resume."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    action = data.get("action")  # 'pause' or 'resume'
    
    if action not in ("pause", "resume"):
        return JsonResponse({"error": "Invalid action"}, status=400)
    
    payload = {action: True}
    try:
        with open(COMMAND_PATH, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
        
    return JsonResponse({"status": action})


def get_stats(request):
    """
    Returns attendance stats from DB.
    
    RULES:
    - Django is READ-ONLY (main.py is the only writer)
    - Short timeout (5s) to avoid blocking
    - All values must be JSON-serializable (no bytes, numpy, datetime objects)
    - is_running is based on CV_PROCESS state (source of truth)
    - active is based on DB session status
    """
    # is_running = CV process is alive (source of truth for engine state)
    is_running = CV_PROCESS is not None and CV_PROCESS.poll() is None
    
    stats = {
        "is_running": is_running,  # Engine process state
        "active": False,           # DB session state
        "session_name": "",
        "present": 0,
        "total": 0,
        "attendees": [],
        "unknowns": []
    }
    
    conn = None
    try:
        # Short-lived read-only connection
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()
        
        # Get active session
        cursor.execute(
            "SELECT id, session_name FROM sessions WHERE status='ACTIVE' ORDER BY id DESC LIMIT 1"
        )
        session = cursor.fetchone()
        
        if not session:
            return JsonResponse(stats)
        
        session_id, session_name = session
        stats["active"] = True
        stats["session_name"] = session_name or ""
        
        # Count Present
        cursor.execute(
            "SELECT COUNT(*) FROM session_attendance WHERE session_id=? AND status='Present'",
            (session_id,)
        )
        stats["present"] = cursor.fetchone()[0] or 0
        
        # Count Total
        cursor.execute(
            "SELECT COUNT(*) FROM session_attendance WHERE session_id=?",
            (session_id,)
        )
        stats["total"] = cursor.fetchone()[0] or 0
        
        # Get Present Students (FULLY QUALIFIED column names to avoid ambiguity)
        cursor.execute("""
            SELECT students.name, students.student_id, session_attendance.last_seen 
            FROM session_attendance 
            JOIN students ON session_attendance.student_id = students.student_id 
            WHERE session_attendance.session_id=? AND session_attendance.status='Present' 
            ORDER BY session_attendance.last_seen DESC
        """, (session_id,))
        
        for row in cursor.fetchall():
            name, student_id, last_seen = row
            stats["attendees"].append({
                "name": str(name) if name else "",
                "id": str(student_id) if student_id else "",
                "last_seen": str(last_seen) if last_seen else ""
            })
        
        # Get Unknown Detections
        cursor.execute("""
            SELECT track_id, timestamp, image_path 
            FROM unknown_detections 
            WHERE session_id=? 
            ORDER BY id DESC LIMIT 5
        """, (session_id,))
        
        for row in cursor.fetchall():
            track_id, timestamp, image_path = row
            
            # Handle track_id carefully (might be bytes from numpy int64)
            if isinstance(track_id, bytes):
                try:
                    track_id = int.from_bytes(track_id, "little")
                except Exception:
                    track_id = 0
            else:
                try:
                    track_id = int(track_id)
                except (ValueError, TypeError):
                    track_id = 0
            
            # Convert image path to URL (MEDIA_ROOT is data/, so strip 'data/' prefix)
            rel_path = ""
            if image_path:
                try:
                    # Handle relative paths like "data/unknown_faces/Session_xxx/..."
                    if image_path.startswith("data/"):
                        rel_path = "/media/" + image_path[5:]  # Strip "data/" prefix
                    # Handle absolute paths
                    elif os.path.isabs(image_path):
                        data_dir = os.path.join(PROJECT_ROOT, "data")
                        if image_path.startswith(str(data_dir)):
                            rel_path = "/media/" + os.path.relpath(image_path, data_dir)
                        else:
                            rel_path = "/media/" + os.path.basename(image_path)
                    else:
                        # Fallback: use path as-is with /media/ prefix
                        rel_path = "/media/" + image_path
                except Exception:
                    rel_path = ""
            
            stats["unknowns"].append({
                "track_id": track_id,
                "time": str(timestamp) if timestamp else "",
                "image": rel_path
            })
        
    except sqlite3.OperationalError as e:
        # Database locked or other SQLite error - return cached/empty stats
        stats["error"] = f"Database busy: {e}"
    except Exception as e:
        stats["error"] = str(e)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
    
    return JsonResponse(stats)


# ============================================
# MP4 PROCESSING ENDPOINTS
# ============================================

def mp4_page(request):
    """Renders the MP4 processing page."""
    # Get available sections
    sections = []
    if os.path.exists(EMBEDDINGS_DIR):
        for f in os.listdir(EMBEDDINGS_DIR):
            if f.endswith(".pkl"):
                sections.append(f.replace(".pkl", ""))
    
    # Check if system is busy
    is_busy = CV_PROCESS is not None and CV_PROCESS.poll() is None
    
    context = {
        "sections": sorted(sections),
        "is_busy": is_busy,
        "busy_type": PROCESS_TYPE if is_busy else None
    }
    return render(request, "dashboard/mp4.html", context)


@csrf_exempt
def upload_mp4(request):
    """Handles MP4 file upload."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)
    
    uploaded_file = request.FILES["file"]
    
    # Validate file type
    if not uploaded_file.name.lower().endswith(".mp4"):
        return JsonResponse({"error": "Only MP4 files are allowed"}, status=400)
    
    # Validate file size (max 500MB)
    if uploaded_file.size > 500 * 1024 * 1024:
        return JsonResponse({"error": "File too large (max 500MB)"}, status=400)
    
    # Ensure uploads directory exists
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    # Save file with timestamp to avoid conflicts
    filename = f"upload_{int(time.time())}_{uploaded_file.name}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    
    try:
        with open(filepath, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        
        return JsonResponse({
            "status": "uploaded",
            "filename": filename,
            "filepath": filepath,
            "size": uploaded_file.size
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def start_mp4(request):
    """Starts MP4 processing with main.py."""
    global CV_PROCESS, PROCESS_TYPE, MP4_PROGRESS
    
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    # Global lock check
    if CV_PROCESS is not None and CV_PROCESS.poll() is None:
        return JsonResponse({"error": f"System busy ({PROCESS_TYPE})"}, status=409)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    filepath = data.get("filepath")
    section = data.get("section")
    session_name = data.get("session", "").strip()
    export_enabled = data.get("export", True)
    
    if not filepath or not os.path.exists(filepath):
        return JsonResponse({"error": "Video file not found"}, status=400)
    
    if not section:
        return JsonResponse({"error": "Section is required"}, status=400)
    
    # Clear old progress file
    try:
        if os.path.exists(MP4_PROGRESS_PATH):
            os.remove(MP4_PROGRESS_PATH)
    except Exception:
        pass
    
    # Reset progress
    MP4_PROGRESS = {"frame": 0, "total": 0, "status": "starting"}
    
    # Construct command
    cmd = [
        os.path.join(PROJECT_ROOT, "venv/bin/python3"),
        os.path.join(PROJECT_ROOT, "main.py"),
        "--section", section,
        "--source", filepath
    ]
    
    if export_enabled:
        cmd.append("--export")
    
    if session_name:
        cmd.extend(["--session", session_name])
    else:
        session_name = f"MP4_{int(time.time())}"
        cmd.extend(["--session", session_name])
    
    try:
        CV_PROCESS = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        PROCESS_TYPE = "mp4"
        return JsonResponse({
            "status": "started",
            "pid": CV_PROCESS.pid,
            "session": session_name
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def stop_mp4(request):
    """Stops MP4 processing."""
    global CV_PROCESS, PROCESS_TYPE, MP4_PROGRESS
    
    # Write stop command
    try:
        with open(COMMAND_PATH, "w") as f:
            json.dump({"stop": True}, f)
    except Exception:
        pass
    
    if CV_PROCESS is None:
        MP4_PROGRESS["status"] = "stopped"
        return JsonResponse({"status": "stopped"})
    
    # Wait for graceful exit
    for _ in range(10):
        if CV_PROCESS.poll() is not None:
            break
        time.sleep(0.5)
    
    if CV_PROCESS.poll() is None:
        CV_PROCESS.terminate()
    
    CV_PROCESS = None
    PROCESS_TYPE = None
    MP4_PROGRESS["status"] = "stopped"
    return JsonResponse({"status": "stopped"})


@csrf_exempt
def pause_mp4(request):
    """Toggles pause/resume for MP4 processing."""
    global MP4_PROGRESS
    
    if CV_PROCESS is None or CV_PROCESS.poll() is not None:
        return JsonResponse({"status": "not_running"})
    
    # Toggle pause/resume
    is_paused = MP4_PROGRESS.get("paused", False)
    
    try:
        with open(COMMAND_PATH, "w") as f:
            if is_paused:
                json.dump({"resume": True}, f)
            else:
                json.dump({"pause": True}, f)
    except Exception:
        pass
    
    # Update local state
    MP4_PROGRESS["paused"] = not is_paused
    
    return JsonResponse({"status": "resumed" if is_paused else "paused", "paused": not is_paused})


def mp4_progress(request):
    """Returns MP4 processing progress."""
    global MP4_PROGRESS
    
    # Check if process is still running
    is_running = CV_PROCESS is not None and CV_PROCESS.poll() is None
    
    # Read progress from file if it exists
    if os.path.exists(MP4_PROGRESS_PATH):
        try:
            with open(MP4_PROGRESS_PATH, "r") as f:
                file_progress = json.load(f)
                MP4_PROGRESS["frame"] = file_progress.get("frame", 0)
                MP4_PROGRESS["total"] = file_progress.get("total", 0)
        except Exception:
            pass
    
    # Update status based on process state
    if not is_running and MP4_PROGRESS["status"] not in ("stopped", "done", "idle"):
        # Process finished
        if MP4_PROGRESS["frame"] > 0 and MP4_PROGRESS["frame"] >= MP4_PROGRESS["total"] - 10:
            MP4_PROGRESS["status"] = "done"
        elif MP4_PROGRESS["frame"] > 0:
            MP4_PROGRESS["status"] = "done"  # Assume done if process exited cleanly
        else:
            MP4_PROGRESS["status"] = "idle"
    elif is_running:
        MP4_PROGRESS["status"] = "processing"
    
    # Calculate percentage
    percentage = 0
    if MP4_PROGRESS["total"] > 0:
        percentage = min(100, int((MP4_PROGRESS["frame"] / MP4_PROGRESS["total"]) * 100))
    
    return JsonResponse({
        "frame": MP4_PROGRESS["frame"],
        "total": MP4_PROGRESS["total"],
        "percentage": percentage,
        "status": MP4_PROGRESS["status"],
        "is_running": is_running
    })


def mp4_results(request):
    """Returns attendance results for the completed MP4 session."""
    conn = None
    results = {
        "session_name": "",
        "present": 0,
        "total": 0,
        "attendees": [],
        "unknowns_count": 0,
        "unknowns": [],
        "export_path": ""
    }
    
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()
        
        # Get most recent closed session (likely the MP4 session)
        cursor.execute("""
            SELECT id, session_name FROM sessions 
            WHERE status='CLOSED' 
            ORDER BY id DESC LIMIT 1
        """)
        session = cursor.fetchone()
        
        if not session:
            return JsonResponse(results)
        
        session_id, session_name = session
        results["session_name"] = session_name or ""
        
        # Check for export file - return actual folder path for user reference
        export_path = os.path.join(PROJECT_ROOT, f"data/exports/{session_name}.mp4")
        if os.path.exists(export_path):
            results["export_path"] = f"data/exports/{session_name}.mp4"
        
        # Count present
        cursor.execute(
            "SELECT COUNT(*) FROM session_attendance WHERE session_id=? AND status='Present'",
            (session_id,)
        )
        results["present"] = cursor.fetchone()[0] or 0
        
        # Count total
        cursor.execute(
            "SELECT COUNT(*) FROM session_attendance WHERE session_id=?",
            (session_id,)
        )
        results["total"] = cursor.fetchone()[0] or 0
        
        # Get attendees
        cursor.execute("""
            SELECT students.name, students.student_id, session_attendance.last_seen
            FROM session_attendance
            JOIN students ON session_attendance.student_id = students.student_id
            WHERE session_attendance.session_id=? AND session_attendance.status='Present'
            ORDER BY students.name
        """, (session_id,))
        
        for row in cursor.fetchall():
            name, student_id, last_seen = row
            results["attendees"].append({
                "name": str(name) if name else "",
                "id": str(student_id) if student_id else "",
                "last_seen": str(last_seen) if last_seen else ""
            })
        
        # Get unknowns with images
        cursor.execute("""
            SELECT track_id, timestamp, image_path 
            FROM unknown_detections 
            WHERE session_id=? 
            ORDER BY id DESC LIMIT 20
        """, (session_id,))
        
        for row in cursor.fetchall():
            track_id, timestamp, image_path = row
            
            # Handle track_id
            if isinstance(track_id, bytes):
                try:
                    track_id = int.from_bytes(track_id, "little")
                except Exception:
                    track_id = 0
            else:
                try:
                    track_id = int(track_id)
                except (ValueError, TypeError):
                    track_id = 0
            
            # Convert image path to URL
            rel_path = ""
            if image_path:
                try:
                    if image_path.startswith("data/"):
                        rel_path = "/media/" + image_path[5:]
                    elif os.path.isabs(image_path):
                        data_dir = os.path.join(PROJECT_ROOT, "data")
                        if image_path.startswith(str(data_dir)):
                            rel_path = "/media/" + os.path.relpath(image_path, data_dir)
                        else:
                            rel_path = "/media/" + os.path.basename(image_path)
                    else:
                        rel_path = "/media/" + image_path
                except Exception:
                    rel_path = ""
            
            results["unknowns"].append({
                "track_id": track_id,
                "time": str(timestamp) if timestamp else "",
                "image": rel_path
            })
        
        results["unknowns_count"] = len(results["unknowns"])
        
    except Exception as e:
        results["error"] = str(e)
    finally:
        if conn:
            conn.close()
    
    return JsonResponse(results)
