import cv2
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
RETRY_COOLDOWN = 1.0  # Seconds before retrying recognition for a track
HEARTBEAT_INTERVAL = 10.0  # Seconds between DB updates for locked IDs

# Shared State
recognition_queue = queue.Queue()
tracked_identities = {}  # {track_id: student_name}
last_recognition_attempt = {}  # {track_id: timestamp}
last_heartbeat_time = {} # {student_name: timestamp}
reid_events = {} # {track_id: start_time} for UI tag

def load_embeddings():
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def compute_similarity(embedding1, embedding2):
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # InsightFace embeddings are usually normalized, but let's be safe.
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def calc_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0
    return intersection_area / union_area

class RecognitionWorker(threading.Thread):
    def __init__(self, session_name):
        super().__init__()
        self.session_name = session_name
        self.daemon = True
        self.running = True
        self.known_embeddings = load_embeddings()
        self.db = DBManager()
        
        # Initialize InsightFace
        print("[Worker] Initializing InsightFace...")
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[Worker] Error initializing InsightFace: {e}")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[Worker] Ready.")

    def run(self):
        while self.running:
            try:
                # Get crop from queue
                track_id, crop = recognition_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Detect face in the person crop
                faces = self.app.get(crop)
                
                if len(faces) == 0:
                    continue
                
                # Assume largest face is the person
                largest_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
                embedding = largest_face.embedding
                
                # Compare with known embeddings
                best_match = None
                max_score = -1.0
                
                for name, known_emb in self.known_embeddings.items():
                    score = compute_similarity(embedding, known_emb)
                    if score > max_score:
                        max_score = score
                        best_match = name
                
                    if max_score > SIMILARITY_THRESHOLD and best_match:
                        # Lock Identity
                        tracked_identities[track_id] = best_match
                        
                        # Re-ID Trigger Logic: 
                        # If this person was seen recently (heartbeat exists), trigger Re-ID tag
                        if best_match in last_heartbeat_time:
                             reid_events[track_id] = time.time()
                             
                        print(f"[Worker] ID Locked: {track_id} -> {best_match} (Score: {max_score:.2f})")
                        
                        # Log Initial Heartbeat
                        self.db.log_heartbeat(best_match, self.session_name)
                        
                        # Set initial heartbeat time
                        last_heartbeat_time[best_match] = time.time()
                else:
                    # Recognition failed or low score
                    pass

            except Exception as e:
                print(f"[Worker] Error processing track {track_id}: {e}")
            finally:
                recognition_queue.task_done()

def main():
    # Session Setup
    session_name = input("Enter Session Name (e.g., Math_101): ").strip()
    if not session_name:
        session_name = f"Session_{int(time.time())}"
        print(f"Defaulting to: {session_name}")

    # Start Worker
    worker = RecognitionWorker(session_name)
    worker.start()

    # YOLO Setup
    print("[Main] Loading YOLO11n...")
    model = YOLO('yolo11n.pt')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Database instance for main thread (if needed, though heartbeats mostly in worker/logic)
    db = DBManager()

    print("[Main] Starting Tracking Loop...")
    try:
        # Loop
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Performance Tuning: Agnostic NMS, Disable Augmentation, Use Custom Tracker
            results = model.track(frame, persist=True, tracker="botsort_custom.yaml", device="mps", vid_stride=3, verbose=False, classes=0, iou=0.5, conf=0.5, agnostic_nms=True, augment=False)
            
            current_tracks = []
            
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                
                # Combine into a list of dicts for easier filtering
                detections = []
                for box, track_id, conf in zip(boxes, ids, confs):
                    detections.append({
                        'box': box,
                        'id': track_id,
                        'conf': conf,
                        'locked': track_id in tracked_identities
                    })
                
                # Overlap Suppression Logic
                keep_indices = set(range(len(detections)))
                for i in range(len(detections)):
                    if i not in keep_indices: continue
                    for j in range(i + 1, len(detections)):
                        if j not in keep_indices: continue
                        
                        det1 = detections[i]
                        det2 = detections[j]
                        
                        iou = calc_iou(det1['box'], det2['box'])
                        if iou > 0.7:
                            # Suppression required
                            drop_idx = -1
                            
                            # Priority 1: Keep Locked ID
                            if det1['locked'] and not det2['locked']:
                                drop_idx = j
                            elif not det1['locked'] and det2['locked']:
                                drop_idx = i
                            # Priority 2: Higher Confidence
                            else:
                                if det1['conf'] >= det2['conf']:
                                    drop_idx = j
                                else:
                                    drop_idx = i
                            
                            if drop_idx != -1:
                                keep_indices.discard(drop_idx)
                                
                # Display remaining tracks
                for idx in keep_indices:
                    det = detections[idx]
                    x1, y1, x2, y2 = map(int, det['box'])
                    track_id = det['id']
                    
                    current_tracks.append(track_id)
                    
                    # Visualization Logic
                    name = tracked_identities.get(track_id)
                    color = (0, 255, 0) if name else (0, 165, 255) # Green if locked, Orange if unknown
                    label = name if name else f"ID:{track_id} Identifying..."
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Re-ID Tag Visualization
                    if track_id in reid_events:
                        if time.time() - reid_events[track_id] < 2.0:
                            # Draw "Re-ID" tag
                            cv2.rectangle(frame, (x1, y1 - 35), (x1 + 60, y1 - 15), (255, 0, 0), -1) # Blue bg
                            cv2.putText(frame, "Re-ID", (x1 + 5, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        else:
                             # Cleanup old event
                             del reid_events[track_id]
                    
                    # Heartbeat & Recognition Logic
                    if name:
                        # Existing lock: Check heartbeat throttle
                        last_hb = last_heartbeat_time.get(name, 0)
                        if time.time() - last_hb > HEARTBEAT_INTERVAL:
                            # Log heartbeat (on main thread or queue? DBManager creates new connection, so Main thread is safe)
                            # Actually, let's keep DB ops off main thread if possible to avoid lag?
                            # DBManager is fast (local sqlite). Let's do it here for simplicity, or spin a quick thread.
                            # Calling it here.
                            threading.Thread(target=db.log_heartbeat, args=(name, session_name), daemon=True).start()
                            last_heartbeat_time[name] = time.time()
                    else:
                        # Unknown ID: Send to recognition?
                        last_attempt = last_recognition_attempt.get(track_id, 0)
                        if time.time() - last_attempt > RETRY_COOLDOWN:
                            # Extract crop
                            # Ensure crop is within bounds
                            h, w, _ = frame.shape
                            cx1, cy1 = max(0, x1), max(0, y1)
                            cx2, cy2 = min(w, x2), min(h, y2)
                            
                            if cx2 > cx1 and cy2 > cy1:
                                crop = frame[cy1:cy2, cx1:cx2].copy()
                                
                                # Virtual Zoom: Upscale small faces (e.g., height < 120px)
                                h_crop, w_crop = crop.shape[:2]
                                if h_crop < 120 or w_crop < 120:
                                    crop = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                                
                                recognition_queue.put((track_id, crop))
                                last_recognition_attempt[track_id] = time.time()

            # Clean up old identities? (Optional, maybe not needed for this session)
            
            # UI Dashboard Overlay
            h_frame, w_frame = frame.shape[:2]
            
            # 1. Semi-transparent Top Bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w_frame, 50), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # 2. Stats
            fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            live_count = len(current_tracks)
            verified_count = sum(1 for tid in current_tracks if tid in tracked_identities)
            
            stats_text = f"FPS: {fps:.1f} | Live: {live_count} | Verified: {verified_count}"
            cv2.putText(frame, stats_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 3. Recording Indicator (Green Dot)
            cv2.circle(frame, (w_frame - 30, 25), 10, (0, 255, 0), -1)
            cv2.putText(frame, "REC", (w_frame - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("MOT Attendance System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        worker.running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Exited.")

if __name__ == "__main__":
    main()
