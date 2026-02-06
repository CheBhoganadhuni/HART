import cv2
import os
import threading
import queue
import time
import numpy as np
import pickle
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from core.db_manager import DBManager

# Constants
EMBEDDINGS_PATH = "data/embeddings.pkl"
SIMILARITY_THRESHOLD = 0.45
RETRY_COOLDOWN = 1.0 
HEARTBEAT_INTERVAL = 10.0 
UNKNOWN_SAVE_THRESHOLD = 15.0 # Increased for stability

# Shared State
recognition_queue = queue.Queue(maxsize=10)
tracked_identities = {} 
last_recognition_attempt = {} 
last_heartbeat_time = {} 
reid_events = {} 
track_start_time = {} 
saved_unknowns = set()
blacklisted_tracks = set() # {track_id} - Ignore for recognition after timeout
last_seen_time = {} # {student_name: timestamp} - For final flush
session_cache = {} # {student_name: embedding} - Priority cache for re-identification
recognition_history = {} # {track_id: [name1, name2, ...]} - Consensus buffer
reid_metadata = {} # {track_id: {"source": "cache/db", "time": timestamp}} - For Blue Tag logic

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
    def __init__(self, session_name):
        super().__init__()
        self.session_name = session_name
        self.daemon = True
        self.running = True
        self.known_embeddings = load_embeddings()
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
                
                # 1. Priority Recognition: Check Session Cache first
                # Strict Session Cache: Higher threshold (0.50)
                for name, cached_emb in session_cache.items():
                    score = np.dot(face.embedding, cached_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(cached_emb))
                    if score > 0.50: # Increased from 0.40
                         max_score, best_match = score, name
                         match_source = "cache"
                         print(f"[Worker] Priority Re-ID: {name} ({score:.2f})")
                         break
                
                # 2. Standard Search: If no cache match, check full DB
                if not best_match:
                    for name, known_emb in self.known_embeddings.items():
                        score = np.dot(face.embedding, known_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(known_emb))
                        if score > max_score:
                            max_score, best_match = score, name
                            match_source = "db"
                
                if max_score > SIMILARITY_THRESHOLD:
                    # Consensus Logic: Require 3 consistent matches
                    if track_id not in recognition_history: recognition_history[track_id] = []
                    recognition_history[track_id].append(best_match)
                    
                    # Keep only last 5 matches
                    if len(recognition_history[track_id]) > 5: recognition_history[track_id].pop(0)

                    # Check for consensus (3 occurrences of same name)
                    recent_matches = recognition_history[track_id]
                    if recent_matches.count(best_match) >= 3:
                        tracked_identities[track_id] = best_match
                        
                        # Store Re-ID metadata for Blue Tag
                        if track_id not in reid_metadata:
                            reid_metadata[track_id] = {
                                "source": match_source,
                                "time": time.time()
                            }

                        # Strict Session Cache Update: Only if score > 0.60
                        if max_score > 0.60:
                            session_cache[best_match] = face.embedding 
                        
                        if best_match in last_heartbeat_time:
                             reid_events[track_id] = time.time()
                        threading.Thread(target=self.db.log_heartbeat, args=(best_match, self.session_name), daemon=True).start()
                        last_heartbeat_time[best_match] = time.time()
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

def main():
    session_name = input("Session Name: ").strip() or f"Session_{int(time.time())}"
    worker = RecognitionWorker(session_name).start()
    
    source = input("Source (0/Path): ").strip()
    is_mp4 = source != '0'
    stream = AsyncVideoStream(source, stride=2 if is_mp4 else 3).start()
    
    model = YOLO('yolo11n.pt')
    session_start = time.time()
    
    try:
        while not stream.stopped or not stream.queue.empty():
            loop_start = time.time()
            frame_idx, frame = stream.read()
            
            # Dynamic Detection: Lower conf to 0.3 for better recall
            results = model.track(frame, persist=True, tracker="botsort_custom.yaml", device="mps", verbose=False, classes=0, agnostic_nms=True, conf=0.3)
            
            current_ids = []
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, tid in zip(boxes, ids):
                    current_ids.append(tid)
                    name = tracked_identities.get(tid)
                    
                    if not name:
                        if tid not in track_start_time: track_start_time[tid] = time.time()
                        elif time.time() - track_start_time[tid] > 5.0:
                             # Unknown Throttling: Blacklist after 5s
                             blacklisted_tracks.add(tid)
                        
                        if time.time() - track_start_time.get(tid, time.time()) > UNKNOWN_SAVE_THRESHOLD:
                            if tid not in saved_unknowns:
                                os.makedirs("data/unknown_faces", exist_ok=True)
                                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                                cv2.imwrite(f"data/unknown_faces/unknown_{session_name}_{tid}.jpg", face)
                                saved_unknowns.add(tid)
                    else:
                        # Update last seen time for known students
                        last_seen_time[name] = time.time()
                    
                    # Visitor Pulse Logic
                    is_pulsing = False
                    if tid in blacklisted_tracks and not name:
                         # Pulse every 10s for 2s
                         pulse_cycle = (time.time() - track_start_time[tid]) % 12.0 # 10s off + 2s on
                         if pulse_cycle > 10.0:
                              is_pulsing = True

                    # Recognition Trigger: Normal OR Pulsing Visitor
                    should_recognize = (not name and tid not in blacklisted_tracks) or is_pulsing
                    
                    if should_recognize and time.time() - last_recognition_attempt.get(tid, 0) > RETRY_COOLDOWN:
                        crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                        if crop.size > 0:
                            if crop.shape[0] < 120: crop = cv2.resize(crop, (0,0), fx=2, fy=2)
                            recognition_queue.put((tid, crop))
                            last_recognition_attempt[tid] = time.time()

                    # UI Drawing
                    color = (0, 255, 0) # Green by default for identified
                    label = name if name else f"ID:{tid} Identifying..."
                    
                    # Blue Tag Logic: Session Re-ID
                    if name and tid in reid_metadata:
                        meta = reid_metadata[tid]
                        if meta['source'] == 'cache' and (time.time() - meta['time'] < 2.0):
                             color = (255, 0, 0) # Blue (BGR)
                             label = f"{name} (Re-ID)"
                    
                    if not name:
                         color = (0, 165, 255) # Orange for identifying

                    # Graceful Unknowns: Visitor Mode
                    if tid in blacklisted_tracks and not name:
                        if is_pulsing:
                             color = (0, 165, 255) # Orange (Pulse Active)
                             label = f"Visitor ID:{tid} (Checking...)"
                        else:
                             color = (192, 192, 192) # Gray
                             label = f"Visitor ID:{tid}"
                    
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(frame, label, (int(box[0]), int(box[1])-10), 1, 1, color, 2)


            # Dashboard
            fps = 1.0 / (time.time() - loop_start)
            elapsed = time.strftime("%M:%S", time.gmtime(time.time() - session_start))
            prog = f" | {int((frame_idx/stream.total_frames)*100)}%" if is_mp4 else ""
            cv2.putText(frame, f"FPS: {fps:.1f} | Live: {len(current_ids)} | {elapsed}{prog}", (20, 30), 1, 1.2, (255,255,255), 2)
            
            cv2.imshow("Attendance Pro", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print(f"Done. Processed in {((time.time()-session_start)/60):.2f}m")
        
        # Final Cleanup: Flush recent heartbeats
        print("Flushing final heartbeats...")
        db = DBManager() # New instance for main thread cleanup
        for name, timestamp in last_seen_time.items():
            if time.time() - timestamp < 5.0:
                print(f"Logging final heartbeat for: {name}")
                db.log_heartbeat(name, session_name)

if __name__ == "__main__": main()