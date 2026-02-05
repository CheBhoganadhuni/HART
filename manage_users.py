import os
import pickle
from core.db_manager import DBManager

EMBEDDINGS_PATH = "data/embeddings.pkl"

def load_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_embeddings(data):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def list_users(db):
    print("\n--- Registered Students ---")
    
    # Get from DB
    db_students = set(db.get_all_students())
    
    # Get from Pickle
    pkl_data = load_embeddings()
    pkl_students = set(pkl_data.keys())
    
    all_students = sorted(list(db_students | pkl_students))
    
    if not all_students:
        print("No students found.")
    else:
        print(f"{'Name':<20} | {'DB':<5} | {'Pickle':<6}")
        print("-" * 35)
        for name in all_students:
            in_db = "Yes" if name in db_students else "No"
            in_pkl = "Yes" if name in pkl_students else "No"
            print(f"{name:<20} | {in_db:<5} | {in_pkl:<6}")
    print("---------------------------\n")

def delete_user(db):
    name = input("Enter name to delete: ").strip()
    if not name:
        return

    # Delete from DB
    db_deleted = db.delete_student(name)
    
    # Delete from Pickle
    data = load_embeddings()
    if name in data:
        del data[name]
        save_embeddings(data)
        pkl_deleted = True
        print(f"[Pickle] Removed '{name}' from embeddings.")
    else:
        pkl_deleted = False
        print(f"[Pickle] '{name}' not found in embeddings.")

    if db_deleted or pkl_deleted:
        print(f"Successfully deleted/cleaned up '{name}'.")
    else:
        print(f"Student '{name}' not found anywhere.")

def list_sessions(db):
    print("\n--- Session Summary ---")
    rows = db.get_sessions_summary()
    if not rows:
        print("No sessions recorded.")
    else:
        print(f"{'Session Name':<30} | {'Students':<8} | {'Start Time'}")
        print("-" * 65)
        for name, count, start_time in rows:
            print(f"{name:<30} | {count:<8} | {start_time}")
    print("-----------------------\n")

def main():
    db = DBManager()
    while True:
        print("\n=== Student Management ===")
        print("1. List Students")
        print("2. Delete Student")
        print("3. View Sessions")
        print("4. Exit")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            list_users(db)
        elif choice == '2':
            delete_user(db)
        elif choice == '3':
            list_sessions(db)
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
