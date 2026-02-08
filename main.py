import cv2
import argparse
import os
import threading
import queue
import time
import numpy as np
import pickle
import json
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from core.db_manager import DBManager

# Constants
EMBEDDINGS_PATH = "data/embeddings.pkl"
SIMILARITY_THRESHOLD = 0.45
RETRY_COOLDOWN = 1.0 
UNKNOWN_SAVE_THRESHOLD = 15.0 # Increased for stability

# Shared State
recognition_queue = queue.Queue(maxsize=10)
tracked_identities = {} 
last_recognition_attempt = {} 
reid_events = {} 
track_start_time = {} 
saved_unknowns = set()
blacklisted_tracks = set() # {track_id} - Ignore for recognition after timeout
last_seen_time = {} # {student_name: timestamp} - For final flush
session_cache = {} # {student_name: embedding} - Priority cache for re-identification
recognition_history = {} # {track_id: [name1, name2, ...]} - Consensus buffer
reid_metadata = {} # {track_id: {"source": "cache/db", "time": timestamp}} - For Blue Tag logic
last_verified_time = {} # {track_id: timestamp} - For Green User Throttling (120s)
last_shadow_time = {} # {track_id: timestamp} - For Shadow Updates (30s)
seen_in_pulse_window = set() # {student_id} - Accumulates seen IDs for 10s pulse

class AsyncVideoStream:
    """Dedicated thread for high-speed frame decoding."""
    def __init__(self, source, stride=1):
        self.cap = cv2.VideoCapture(source if source != '0' else 0)
        self.stride = stride
        self.queue = queue.Queue(maxsize=64)
        self.stopped = False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        frame_idx = 0
        while not self.stopped:
            if not self.queue.full():
                # Skip frames physically for MP4 speedup
                for _ in range(self.stride - 1):
                    self.cap.grab()
                
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    return

                self.queue.put((frame_idx, frame))
                frame_idx += self.stride
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

class RecognitionWorker(threading.Thread):
    def __init__(self, session_name, known_embeddings, id_to_name):
        super().__init__()
        self.session_name = session_name
        self.daemon = True
        self.running = True
        self.known_embeddings = known_embeddings
        self.id_to_name = id_to_name # {student_id: name}
        self.db = DBManager()
        
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def run(self):
        while self.running:
            try:
                track_id, crop = recognition_queue.get(timeout=1.0)
            except queue.Empty: continue

            try:
                faces = self.app.get(crop)
                if not faces: continue
                
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                best_match, max_score = None, -1.0
                match_source = "db" # 'cache' or 'db'
                
                # 1. Priority Recognition: Check Session Cache (Multi-Vector)
                # Strict Session Cache: Higher threshold (0.50)
                # cache keys are student_ids
                for student_id, cached_embs in session_cache.items():
                    # Handle legacy cache format (just in case) or new list format
                    if not isinstance(cached_embs, list): cached_embs = [cached_embs]
                    
                    # Check all embeddings in the diversity list
                    for cached_emb in cached_embs:
                        score = np.dot(face.embedding, cached_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(cached_emb))
                        if score > 0.50: # Increased from 0.40
                             if score > max_score:
                                 max_score, best_match = score, student_id
                                 match_source = "cache"
                
                # 2. Standard Search: If no cache match, check full DB
                if not best_match:
                    for student_id, known_emb in self.known_embeddings.items():
                        score = np.dot(face.embedding, known_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(known_emb))
                        if score > max_score:
                            max_score, best_match = score, student_id
                            match_source = "db"
                
                if max_score > SIMILARITY_THRESHOLD:
                    # Consensus Logic: Adaptive (1 for Cache, 2 for DB)
                    if track_id not in recognition_history: recognition_history[track_id] = []
                    recognition_history[track_id].append(best_match)
                    
                    # Keep only last 5 matches
                    if len(recognition_history[track_id]) > 5: recognition_history[track_id].pop(0)

                    # Check for consensus
                    recent_matches = recognition_history[track_id]
                    match_count = recent_matches.count(best_match)
                    
                    required_consensus = 1 if match_source == 'cache' else 2
                    
                    if match_count >= required_consensus:
                        tracked_identities[track_id] = best_match
                        
                        # Store Re-ID metadata for Blue Tag
                        if track_id not in reid_metadata:
                            reid_metadata[track_id] = {
                                "source": match_source,
                                "time": time.time()
                            }

                        # Diversity-Aware Session Cache Update
                        # Quality Gate: det_score > 0.6 AND Size > 60x60
                        box = face.bbox
                        h, w = box[3]-box[1], box[2]-box[0]
                        if max_score > 0.65 and face.det_score > 0.6 and h > 60 and w > 60:
                            current_embs = session_cache.get(best_match, [])
                            if not isinstance(current_embs, list): current_embs = [current_embs]
                            
                            # Redundancy Check: Is this new face > 0.90 similar to any we already have?
                            is_redundant = False
                            most_similar_idx = -1
                            highest_sim = -1.0
                            
                            for idx, existing_emb in enumerate(current_embs):
                                sim = np.dot(face.embedding, existing_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(existing_emb))
                                if sim > highest_sim:
                                    highest_sim = sim
                                    most_similar_idx = idx
                                if sim > 0.90:
                                    is_redundant = True
                            
                            if not is_redundant:
                                # Cache Expansion: Max Size increased to 5
                                if len(current_embs) < 5:
                                    current_embs.append(face.embedding)
                                    print(f"[Worker] [Cache] Added {best_match} (Count: {len(current_embs)}) -> New Angle")
                                else:
                                    # Replace the one most similar to the new one (refining the cluster)
                                    if most_similar_idx != -1:
                                        current_embs[most_similar_idx] = face.embedding
                                        print(f"[Worker] [Cache] Updating {best_match} Idx {most_similar_idx} (Sim: {highest_sim:.2f}) -> Diverse Pose Captured")
                                
                                session_cache[best_match] = current_embs
                            else:
                                 print(f"[Worker] [Cache] Skipped {best_match} - Too Similar (Sim: {highest_sim:.2f})")
                        else:
                             if max_score > 0.65:
                                  print(f"[Worker] [Cache] Skipped {best_match} - Low Quality (Score: {face.det_score:.2f} Size: {h}x{w})")

                        # Update Verification Timestamp (for Throttling)
                        last_verified_time[track_id] = time.time()
            except Exception as e: print(f"Worker Error: {e}")
            finally: recognition_queue.task_done()

def load_embeddings():
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f: return pickle.load(f)
    except: return {}

def calc_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

# IPC Paths
COMMAND_PATH = "data/commands.json"
STREAM_PATH = "data/live_stream.jpg"

def poll_commands():
    """Checks for external commands from the web dashboard.
    
    IMPORTANT: Commands are single-shot. After reading, the file is cleared
    to prevent the same command from being processed every loop iteration.
    """
    if not os.path.exists(COMMAND_PATH): return None
    try:
        with open(COMMAND_PATH, "r") as f:
            cmd = json.load(f)
        
        # Clear the command file after reading (prevents re-processing)
        try: os.remove(COMMAND_PATH)
        except: pass
        
        if cmd.get("stop"):
            return "stop"
        if cmd.get("pause"):
            return "pause"
        if cmd.get("resume"):
            return "resume"
    except Exception:
        pass
    return None

def write_live_stream(frame):
    """Writes the current frame to a file for MJPEG streaming.
    
    Uses cv2.imencode to encode JPEG in memory, then atomic file write.
    This avoids the ".jpg.tmp" extension issue with cv2.imwrite.
    """
    try:
        # Ensure data directory exists
        stream_dir = os.path.dirname(STREAM_PATH)
        if stream_dir and not os.path.exists(stream_dir):
            os.makedirs(stream_dir, exist_ok=True)
        
        # Encode frame to JPEG in memory
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return
        
        # Write bytes atomically: temp file then rename
        temp_path = STREAM_PATH + "_tmp"  # "data/live_stream.jpg_tmp"
        with open(temp_path, 'wb') as f:
            f.write(buffer.tobytes())
        os.replace(temp_path, STREAM_PATH)
    except Exception as e:
        # Silently skip - don't spam logs every frame
        pass

def list_available_sections():
    sections = []
    if os.path.exists("data/embeddings"):
        for f in os.listdir("data/embeddings"):
            if f.endswith(".pkl"):
                sections.append(f.replace(".pkl", ""))
    return sorted(sections)

def main():
    parser = argparse.ArgumentParser(description="Modular Attendance System")
    parser.add_argument('--session', type=str, help='Session Name')
    parser.add_argument('--section', type=str, help='Section Name (Required for Modular Loading)')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam or path)')
    parser.add_argument('--export', action='store_true', help='Enable video export')
    parser.add_argument('--cache', type=str, help='Path to selective cache pkl to load (Headstart)')
    args = parser.parse_args()

    # Section Validation
    available_sections = list_available_sections()
    
    if not args.section:
        print("\n[Error] Section name is required.")
        if available_sections:
            print(f"Available Sections: {', '.join(available_sections)}")
        else:
            print("No sections found. Please run 'register_student.py' first.")
        print("Usage: python3 main.py --section <SECTION_NAME>\n")
        return

    section_name = args.section
    
    # Check if section exists
    section_file = f"data/embeddings/{section_name}.pkl"
    if not os.path.exists(section_file):
         print(f"\n[Error] Section '{section_name}' does not exist.")
         if available_sections:
            print(f"Did you mean? {', '.join(available_sections)}")
         return

    session_name = args.session or f"Session_{int(time.time())}"
    
    print(f"[System] Starting Session: {session_name} for Section: {section_name}")
    
    # 1. Modular Loading: Load ONLY this section's embeddings
    known_embeddings = {}
    try:
        with open(section_file, 'rb') as f: known_embeddings = pickle.load(f)
        # keys are now student_ids
        print(f"[System] Loaded {len(known_embeddings)} student signatures from {section_name}.")
    except Exception as e:
        print(f"[System] Failed to load section embeddings: {e}")
        return

    # Selective Cache Loading (Headstart)
    if args.cache:
        if os.path.exists(args.cache):
            try:
                with open(args.cache, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    session_cache.update(loaded_cache)
                print(f"[System] Headstart enabled: Loaded cache from {args.cache}")
            except Exception as e:
                print(f"[System] Failed to load cache: {e}")
        else:
            print(f"[System] Warning: Cache file not found at {args.cache}")
    
    # Initialize DB Session
    db = DBManager()
    session_id = db.start_session(session_name, section_name)
    print(f"[System] DB Session Started (ID: {session_id})")
    
    # Load ID -> Name Mapping for Display
    # db.get_section_students returns dict {id: name}
    id_to_name = db.get_section_students(section_name)
    
    worker = RecognitionWorker(session_name, known_embeddings, id_to_name).start() # Pass specific embeddings & mapping
    
    # ... stream setup ...
    source = args.source
    is_mp4 = source != '0'
    stream = AsyncVideoStream(source, stride=2 if is_mp4 else 3).start()
    
    # ... model setup ...
    model = YOLO('yolo11n.pt')
    session_start = time.time()
    
    # Global Pulse Timers
    last_global_pulse = time.time()
    
    is_paused = False
    
    # Export Setup

    # Export Setup
    video_writer = None
    if args.export:
        os.makedirs("data/exports", exist_ok=True)
        export_path = f"data/exports/{session_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Wait for first frame to get size
        while stream.queue.empty() and not stream.stopped:
            time.sleep(0.1)
        if not stream.queue.empty():
             # Peek at frame size
             _, sample_frame = stream.queue.queue[0]
             h, w = sample_frame.shape[:2]
             video_writer = cv2.VideoWriter(export_path, fourcc, 30.0 if is_mp4 else 30.0, (w, h))
             print(f"[System] Export enabled: Recording to {export_path}")

    try:
        while not stream.stopped or not stream.queue.empty():
            loop_start = time.time()
            frame_idx, frame = stream.read()
            
            # Global Heartbeat Pulse (Every 10s)
            if time.time() - last_global_pulse > 10.0:
                # Pulse Logic: Only mark students seen in the accumulation window
                present_ids = list(seen_in_pulse_window)
                
                # Sync to DB (Bulk Update)
                threading.Thread(target=db.global_heartbeat_sync, args=(session_id, present_ids), daemon=True).start()
                
                # Reset Window
                seen_in_pulse_window.clear()
                last_global_pulse = time.time()
            
            if not is_paused:
                # Dynamic Detection: Agnostic NMS True, IOU=0.5 to prevent ghosting
                results = model.track(frame, persist=True, tracker="botsort_custom.yaml", device="mps", verbose=False, classes=0, agnostic_nms=True, conf=0.15, iou=0.5)
                
                current_ids = []
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, tid, conf in zip(boxes, ids, confs):
                        # Regional Confidence Tuning
                        # Top 35% (Back Row): Accept all > 0.15 (Global)
                        # Bottom 65% (Front Row): Require > 0.30 to filter noise
                        y_center = (box[1] + box[3]) / 2
                        if y_center > frame.shape[0] * 0.35 and conf < 0.30:
                            continue
                            
                        current_ids.append(tid)

                        # Accumulate for Heartbeat Pulse
                        if tid in tracked_identities and tracked_identities[tid]:
                             seen_in_pulse_window.add(tracked_identities[tid])

                        name = tracked_identities.get(tid)
                        
                        if not name:
                            if tid not in track_start_time: track_start_time[tid] = time.time()
                            elif time.time() - track_start_time[tid] > 5.0:
                                 # Unknown Throttling: Blacklist after 5s
                                 blacklisted_tracks.add(tid)
                            
                            if time.time() - track_start_time.get(tid, time.time()) > UNKNOWN_SAVE_THRESHOLD:
                                if tid not in saved_unknowns:
                                    save_dir = f"data/unknown_faces/{session_name}"
                                    os.makedirs(save_dir, exist_ok=True)
                                    
                                    face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                                    img_path = f"{save_dir}/unknown_{tid}_{int(time.time())}.jpg"
                                    cv2.imwrite(img_path, face)
                                    
                                    # Log to DB
                                    threading.Thread(target=db.log_unknown_detection, args=(session_id, session_name, tid, img_path), daemon=True).start()
                                    
                                    saved_unknowns.add(tid)
                        else:
                            # Update last seen time for known students
                            last_seen_time[name] = time.time()
                        
                        # [Removed Duplicate Drawing Block]
                        
                        # Visitor Pulse Logic
                        is_pulsing = False
                        if tid in blacklisted_tracks and not name:
                             # Pulse every 10s for 2s
                             pulse_cycle = (time.time() - track_start_time[tid]) % 12.0 # 10s off + 2s on
                             if pulse_cycle > 10.0:
                                  is_pulsing = True

                        # Green User Throttling Logic
                        should_reverify = False
                        is_confirming = False
                        if name:
                            # Re-verify every 120s (Fallback / Heartbeat)
                            time_since_verify = time.time() - last_verified_time.get(tid, 0)
                            if time_since_verify > 120.0:
                                 should_reverify = True
                                 is_confirming = True
                        
                        # Shadow Update Logic (Adaptive Memory Scaling - 30s)
                        should_shadow = False
                        if name and not should_reverify:
                            if time.time() - last_shadow_time.get(tid, 0) > 30.0:
                                 should_shadow = True

                        # Recognition Trigger: Normal OR Pulsing Visitor OR Green Re-Verification OR Shadow Update
                        should_recognize = (not name and tid not in blacklisted_tracks) or is_pulsing or should_reverify or should_shadow
                        
                        if should_recognize and time.time() - last_recognition_attempt.get(tid, 0) > RETRY_COOLDOWN:
                            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                            # Only resize/queue if crop valid AND queue not full (prevent UI freeze)
                            if crop.size > 0:
                                if crop.shape[0] < 120: crop = cv2.resize(crop, (0,0), fx=2, fy=2)
                                
                                if not recognition_queue.full():
                                    recognition_queue.put((tid, crop))
                                    last_recognition_attempt[tid] = time.time()
                                    if should_shadow: last_shadow_time[tid] = time.time()

                        # UI Drawing
                        # UI Drawing (Consolidated)
                        color = (0, 165, 255) # Orange (Identifying)
                        label = f"ID:{tid} Identifying..."

                        if name:
                            display_name = id_to_name.get(tracked_identities[tid], name)
                            color = (0, 255, 0) # Green (Verified)
                            label = display_name
                            
                            # Blue Tag: Re-ID (< 6s)
                            if tid in reid_metadata and (time.time() - reid_metadata[tid]['time'] < 6.0):
                                 color = (255, 0, 0) # Blue
                                 label = f"{display_name} (Re-ID)"
                            
                            # Confirming
                            if is_confirming:
                                 label = f"{display_name} (Confirming...)"
                        
                        elif tid in blacklisted_tracks:
                            # Visitor Logic
                            color = (192, 192, 192) # Gray
                            label = f"Visitor ID:{tid}"
                            if is_pulsing:
                                 color = (0, 165, 255) # Orange
                                 label = f"Visitor ID:{tid} (Checking...)"
                        
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        
                        # Label Logic: Ensure no overlap and stay on screen
                        text_size = cv2.getTextSize(label, 0, 0.6, 2)[0]
                        lbl_x = int(box[0])
                        lbl_y = int(box[1]) - 10
                        if lbl_y < 20: lbl_y = int(box[1]) + 20 # Push down if too high
                        
                        cv2.putText(frame, label, (lbl_x, lbl_y), 0, 0.6, color, 2)


            # Dashboard (Top-Left, Always on top)
            fps = 1.0 / (time.time() - loop_start)
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - session_start))
            prog = f" | {int((frame_idx/stream.total_frames)*100)}%" if is_mp4 else ""
            status_text = f"FPS: {fps:.1f} | Live: {len(current_ids)} | Time: {elapsed}{prog}"
            
            # IPC: Check for commands
            cmd = poll_commands()
            if cmd == "stop":
                print("[System] Stop command received.")
                break
            elif cmd == "pause":
                is_paused = True
                print("[System] Paused by web dashboard.")
            elif cmd == "resume":
                is_paused = False
                print("[System] Resumed by web dashboard.")

            if is_paused:
                status_text += " | PAUSED"
                cv2.putText(frame, "SYSTEM PAUSED (Press 'p' to Resume)", (frame.shape[1]//2 - 200, frame.shape[0]//2), 0, 1.0, (0, 0, 255), 3)

            cv2.putText(frame, status_text, (20, 30), 0, 0.7, (255, 255, 255), 2)
            # Controls text removed - now handled in web UI
            
            if video_writer:
                video_writer.write(frame)

            # IPC: Write Stream
            write_live_stream(frame)
            
            # IPC: Write MP4 progress (every 30 frames to reduce I/O)
            if is_mp4 and frame_idx % 30 == 0:
                try:
                    with open("data/mp4_progress.json", "w") as pf:
                        json.dump({"frame": frame_idx, "total": stream.total_frames}, pf)
                except Exception:
                    pass

            # cv2.imshow("Attendance Pro", frame) # Disabled for web-only mode
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): is_paused = not is_paused

    finally:
        stream.stop()
        
        # Close Session in DB
        try:
            db.close_session(session_id)
        except Exception as e:
            print(f"[System] Failed to close session: {e}")

        if video_writer:
            video_writer.release()
            print("[System] Export saved.")
            
        cv2.destroyAllWindows()
        print(f"Done. Processed in {((time.time()-session_start)/60):.2f}m")
        
        # Persistent Cache Export
        try:
            os.makedirs("data/persistent_cache", exist_ok=True)
            # Modular Cache Name: cache_[section]_[session]_[timestamp].pkl
            cache_name = f"cache_{section_name}_{session_name}_{int(time.time())}.pkl"
            cache_path = os.path.join("data/persistent_cache", cache_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(session_cache, f)
            print(f"[System] Section cache exported to {cache_path}")
        except Exception as e:
            print(f"[System] Failed to export cache: {e}")
        
        # Final Cleanup: Flush recent heartbeats (Wait... Heartbeats are now Global Pulse)
        # We can do one final pulse here
        print("Flushing final pulse...")
        try:
             present_students = [name for tid, name in tracked_identities.items() if name]
             db.global_heartbeat_sync(session_id, present_students)
        except: pass
        
        # Clean up live stream file to avoid stale frames on next session
        try:
            if os.path.exists(STREAM_PATH):
                os.remove(STREAM_PATH)
        except: pass


if __name__ == "__main__": main()