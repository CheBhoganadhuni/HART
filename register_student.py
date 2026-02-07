import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import os
from core.db_manager import DBManager

# Constants
EMBEDDINGS_PATH = "data/embeddings.pkl"
TOTAL_FRAMES = 100
MOVEMENT_THRESHOLD = 1.5 

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception: return {}
    return {}

def save_embeddings(data):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def compute_sim(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def draw_clean_dots(img, lmk):
    """Draws only the 106 landmark dots for a clean look."""
    for point in lmk:
        cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1, cv2.LINE_AA)

def register_student():
    db = DBManager()
    
    # 1. Ask for Section First
    try:
        print("\n=== Biometric Registration ===")
        print("Controls in registration window: 'p' to Pause/Resume | 'r' to Restart | 'q' to Quit")
        print("Please ensure:")
        print("1. You are in a well-lit environment.")
        print("2. Your face is clearly visible (no masks/sunglasses).")
        print("3. No one else is in the camera view.")
        print("--------------------------------")
        
        section = input("Enter Section Name (e.g., CSE_A): ").strip()
        if not section: 
            print("Section name is required.")
            return

        # 2. Ask for Student ID (Unique Identifier)
        student_id = input("Enter Student ID (e.g., CSE001): ").strip()
        if not student_id: 
            print("Student ID is required.")
            return

        # Pre-Check: Prevent Duplicate ID
        # Check in DB
        if db.get_student_by_id(student_id): # We need to add this method to DB or just query
             print(f"\n[ERROR] Student ID '{student_id}' already exists in database. Aborting.")
             return
             
        # Check in Pickle (Double safety)
        section_file = f"data/embeddings/{section}.pkl"
        existing_data = {}
        if os.path.exists(section_file):
            try:
                with open(section_file, 'rb') as f: existing_data = pickle.load(f)
                if student_id in existing_data:
                     print(f"\n[ERROR] Student ID '{student_id}' already exists in section file. Aborting.")
                     return
            except: pass

        # 3. Ask for Student Name (Display Only)
        name = input("Enter Student Name: ").strip()
        if not name: return
        
    except KeyboardInterrupt:
        print("\n[System] Registration cancelled.")
        return
    
    # 4. Load Section-Specific Embeddings
    # Structure: {student_id: embedding} 
    # Note: We are migrating from {name: emb} to {id: emb}. 
    # If old file exists, it might have names as keys. This might cause issues if not handled.
    # For now, let's assume we are starting fresh or user handles migration.
    section_file = f"data/embeddings/{section}.pkl"
    existing_data = {}
    if os.path.exists(section_file):
        try:
            with open(section_file, 'rb') as f: existing_data = pickle.load(f)
        except: pass
    
    app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    captured_embeddings = []
    prev_lmk = None
    
    is_paused = False
    pause_reason = None
    
    print(f"Starting Scan for {name} ({student_id}) in {section}...")
    print("Controls: 'p' to Pause/Resume | 'r' to Restart | 'q' to Quit")
    
    try:
        while len(captured_embeddings) < TOTAL_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            
            h, w, _ = frame.shape
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            faces = app.get(frame)
            
            # User Feedback & Controls
            status_color = (0, 255, 0)
            instruction = "ROTATE HEAD SLOWLY"
            
            if is_paused:
                if pause_reason:
                    instruction = f"{pause_reason} (Press 'r' to Resume)"
                    status_color = (0, 0, 255) # Red for Error Pause
                else:
                    instruction = "PAUSED (Press 'p' to Resume)"
                    status_color = (0, 165, 255) # Orange for Manual Pause
            
            elif len(faces) > 1:
                # Force Pause and Reset
                captured_embeddings = []
                prev_lmk = None
                is_paused = True
                pause_reason = "MULTIPLE FACES! RESETTING..."
                instruction = pause_reason
                status_color = (0, 0, 255)
                
            elif faces:
                # Normal Capture Logic
                pause_reason = None # Clear reason if we are back to normal (though we are paused, so this branch won't hit until resumed)
                
                face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                curr_lmk = face.landmark_2d_106
                draw_clean_dots(display, curr_lmk)
                
                should_capture = False
                if prev_lmk is None:
                    should_capture = True
                else:
                    dist = np.mean(np.linalg.norm(curr_lmk - prev_lmk, axis=1))
                    if dist > MOVEMENT_THRESHOLD:
                        should_capture = True
                
                if should_capture:
                    captured_embeddings.append(face.embedding)
                    prev_lmk = curr_lmk.copy()
                else:
                    status_color = (0, 165, 255) # Orange
            else:
                 # No face
                 instruction = "NO FACE DETECTED"
                 status_color = (0, 0, 255)

            # Draw UI
            progress = int((len(captured_embeddings) / TOTAL_FRAMES) * 100)
            cv2.rectangle(display, (50, h - 50), (w - 50, h - 30), (40, 40, 40), -1)
            cv2.rectangle(display, (50, h - 50), (50 + int((w - 100) * (progress/100)), h - 30), status_color, -1)
            cv2.putText(display, f"Scan Progress: {progress}%", (50, h - 65), 1, 1.2, (255, 255, 255), 1)
            cv2.putText(display, f"ID: {student_id}", (w//2 - 60, 50), 1, 1.0, (200, 200, 200), 2)
            cv2.putText(display, instruction, (w//2 - 140, 90), 1, 1.5, status_color, 2)
            cv2.putText(display, "'p': Pause | 'r': Restart | 'q': Quit", (50, h - 10), 1, 1.0, (200, 200, 200), 1)

            cv2.imshow("Biometric Registration", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("\n[System] Registration cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                captured_embeddings = []
                prev_lmk = None
                is_paused = False
                pause_reason = None
                print("[System] Restarting scan...")
            elif key == ord('p'):
                is_paused = not is_paused
                print(f"[System] {'Paused' if is_paused else 'Resumed'} scan.")
                
    except KeyboardInterrupt:
        print("\n[System] Registration cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        return
 
    if len(captured_embeddings) >= TOTAL_FRAMES:
        mean_emb = np.mean(captured_embeddings, axis=0)
        
        # Check against existing in this section
        for ex_id, ex_emb in existing_data.items():
            if compute_sim(mean_emb, ex_emb) > 0.75:
                print(f"\n[ERROR] Match found with ID '{ex_id}' in {section}. Registration Aborted.")
                return
 
        existing_data[student_id] = mean_emb
        
        # Save to Section File
        os.makedirs("data/embeddings", exist_ok=True)
        with open(section_file, 'wb') as f:
            pickle.dump(existing_data, f)
            
        db.register_student(name, section, student_id)
        print(f"\nSuccess! '{name}' ({student_id}) registered in '{section}'.")
    
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    register_student()