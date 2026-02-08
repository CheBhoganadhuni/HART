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

# Paths
PROJECT_ROOT = settings.BASE_DIR.parent
COMMAND_PATH = os.path.join(PROJECT_ROOT, "data/commands.json")
STREAM_PATH = os.path.join(PROJECT_ROOT, "data/live_stream.jpg")
DB_PATH = os.path.join(PROJECT_ROOT, "data/attendance.db")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "data/embeddings")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data/persistent_cache")


def index(request):
    """Renders the main dashboard."""
    # Get available sections
    sections = []
    if os.path.exists(EMBEDDINGS_DIR):
        for f in os.listdir(EMBEDDINGS_DIR):
            if f.endswith(".pkl"):
                sections.append(f.replace(".pkl", ""))
    
    # Get available cache files
    caches = []
    if os.path.exists(CACHE_DIR):
        for f in sorted(os.listdir(CACHE_DIR), reverse=True):
            if f.startswith("cache_") and f.endswith(".pkl"):
                caches.append(f)
    
    context = {
        "sections": sorted(sections),
        "caches": caches[:10],  # Limit to 10 most recent
        "is_running": CV_PROCESS is not None and CV_PROCESS.poll() is None
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


@csrf_exempt
def start_session(request):
    """Starts the main.py subprocess."""
    global CV_PROCESS, SESSION_METADATA
    
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if CV_PROCESS is not None and CV_PROCESS.poll() is None:
        return JsonResponse({"status": "running", "message": "Session already active"})
    
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
        SESSION_METADATA = {"section": section, "start_time": time.time()}
        return JsonResponse({"status": "started", "pid": CV_PROCESS.pid})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def stop_session(request):
    """Sends stop signal to main.py."""
    global CV_PROCESS
    
    # Write Stop Command
    try:
        with open(COMMAND_PATH, "w") as f:
            json.dump({"stop": True}, f)
    except Exception:
        pass
    
    # Wait for process to exit gracefully
    if CV_PROCESS:
        for _ in range(10):  # 5 seconds total
            if CV_PROCESS.poll() is not None:
                break
            time.sleep(0.5)
        
        if CV_PROCESS.poll() is None:
            CV_PROCESS.terminate()
    
    CV_PROCESS = None
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
            
            # Convert image path to relative URL (relative to MEDIA_ROOT which is data/)
            rel_path = ""
            if image_path:
                try:
                    # MEDIA_ROOT is PROJECT_ROOT/data, so make path relative to that
                    data_dir = os.path.join(PROJECT_ROOT, "data")
                    if image_path.startswith(str(data_dir)):
                        rel_path = "/media/" + os.path.relpath(image_path, data_dir)
                    elif "unknown_faces" in image_path:
                        # Fallback: just use the last part after unknown_faces/
                        rel_path = "/media/unknown_faces/" + os.path.basename(image_path)
                    else:
                        rel_path = "/media/" + os.path.basename(image_path)
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
