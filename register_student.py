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
    name = input("Enter Student Name: ").strip()
    if not name: return

    existing_data = load_embeddings()
    
    app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    cap = cv2.VideoCapture(0)
    captured_embeddings = []
    prev_lmk = None
    
    print(f"Starting Smart Biometric Scan for {name}...")
    
    while len(captured_embeddings) < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        
        # Define dimensions to fix NameError
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        faces = app.get(frame)
        
        if faces:
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
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)

            progress = int((len(captured_embeddings) / TOTAL_FRAMES) * 100)
            cv2.rectangle(display, (50, h - 50), (w - 50, h - 30), (40, 40, 40), -1)
            cv2.rectangle(display, (50, h - 50), (50 + int((w - 100) * (progress/100)), h - 30), color, -1)
            cv2.putText(display, f"Scan Progress: {progress}%", (50, h - 65), 1, 1.2, (255, 255, 255), 1)
            cv2.putText(display, "ROTATE HEAD SLOWLY", (w//2 - 140, 50), 1, 1.5, color, 2)

        cv2.imshow("Biometric Registration", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if len(captured_embeddings) >= TOTAL_FRAMES:
        mean_emb = np.mean(captured_embeddings, axis=0)
        
        for ex_name, ex_emb in existing_data.items():
            if compute_sim(mean_emb, ex_emb) > 0.75:
                print(f"\n[ERROR] Match found with '{ex_name}'. Registration Aborted.")
                return

        existing_data[name] = mean_emb
        save_embeddings(existing_data)
        db.register_student(name)
        print(f"\nSuccess! Registration complete for {name}.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_student()