import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
import os
import sys
from core.db_manager import DBManager

# Constants
EMBEDDINGS_PATH = "data/embeddings.pkl"
FRAMES_TO_CAPTURE = 100

def get_face_analysis():
    """Initializes InsightFace with CoreML or CPU."""
    # Note: On macOS with M-series chips, CoreMLExecutionProvider is the optimal choice if available.
    # Otherwise, CPUExecutionProvider is standard.
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    try:
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"Error initializing FaceAnalysis: {e}")
        print("Falling back to CPUExecutionProvider only.")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(data):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def register_student():
    db = DBManager()
    
    student_name = input("Enter Student Name: ").strip()
    if not student_name:
        print("Name cannot be empty.")
        return

    # Check if student exists
    data = load_embeddings()
    if student_name in data:
        print(f"Warning: Student '{student_name}' already exists.")
        choice = input("Overwrite? (y/N): ").lower()
        if choice != 'y':
            print("Registration cancelled.")
            return

    print(f"Initializing Camera and Model for {student_name}...")
    app = get_face_analysis()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    embeddings = []
    frames_captured = 0
    
    print("Please rotate your head slowly. Capturing 100 frames...")
    
    while frames_captured < FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display feedback
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frames: {frames_captured}/{FRAMES_TO_CAPTURE}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Registration", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break
        
        # Detection
        faces = app.get(frame)
        
        if len(faces) > 0:
            # Sort by area (h*w) to get the largest face, assuming it's the student
            largest_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
            
            embeddings.append(largest_face.embedding)
            frames_captured += 1
            print(f"Captured {frames_captured}/{FRAMES_TO_CAPTURE}", end='\r')
        else:
            pass # No face detected
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(embeddings) > 0:
        # Calculate Mean Embedding (Centroid)
        # embedding shape is (512,) usually for ArcFace
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Load existing embeddings
        data = load_embeddings()
        
        # Add to dictionary
        data[student_name] = avg_embedding
        
        # Save pickle
        save_embeddings(data)
        print(f"\nSaved {len(embeddings)} frames. Average embedding stored.")
        
        # Register in DB
        db.register_student(student_name)
        
        print("Student successfully registered!")
    else:
        print("\nNo face data captured. Registration failed.")

if __name__ == "__main__":
    register_student()
