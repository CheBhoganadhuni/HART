import os
import pickle
from core.db_manager import DBManager
from datetime import datetime

EMBEDDINGS_PATH = "data/embeddings.pkl"

def load_embeddings():
    """Scans data/embeddings/ for all .pkl files and merges them."""
    combined_data = {}
    if os.path.exists("data/embeddings"):
        for f in os.listdir("data/embeddings"):
            if f.endswith(".pkl"):
                try:
                    with open(os.path.join("data/embeddings", f), 'rb') as pkl:
                        data = pickle.load(pkl)
                        combined_data.update(data)
                except: pass
    return combined_data

def save_embeddings(data):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def list_users(db):
    print("\n--- Registered Students ---")
    
    # Get from DB: List of (name, section, student_id)
    try:
        db_students = db.get_all_students_with_section()
    except AttributeError:
        print("Error: DBManager missing 'get_all_students_with_section'")
        return

    # Organize by Section
    students_by_section = {}
    db_ids = set()
    for name, section, student_id in db_students:
        if section not in students_by_section: students_by_section[section] = []
        students_by_section[section].append((name, student_id))
        db_ids.add(student_id)
    
    # Get from Pickle (All sections)
    pkl_ids = set()
    pkl_id_to_name = {} # Try to guess name if possible? No, keys are IDs now.
    # Check all Section PKLs in data/embeddings/
    if os.path.exists("data/embeddings"):
        for f in os.listdir("data/embeddings"):
            if f.endswith(".pkl"):
                try:
                    with open(os.path.join("data/embeddings", f), 'rb') as pkl:
                        data = pickle.load(pkl)
                        pkl_ids.update(data.keys())
                except: pass

    if not db_students and not pkl_ids:
        print("No students found.")
    else:
        print(f"{'Section':<10} | {'Student ID':<15} | {'Name':<20} | {'Status'}")
        print("-" * 65)
        
        # Print DB Students
        for section, students in students_by_section.items():
             for name, student_id in students:
                sid_str = str(student_id) if student_id else "N/A"
                status = "DB+PKL" if sid_str in pkl_ids else "DB Only"
                print(f"{section:<10} | {sid_str:<15} | {name:<20} | {status}")
        
        # Print Orphaned Pickle Students (IDs only)
        orphans = pkl_ids - db_ids
        if orphans:
             print("-" * 65)
             print("Found in Pickle but NOT in DB (Orphans - IDs Only):")
             for sid in orphans:
                 print(f"{'Unknown':<10} | {str(sid):<15} | {'Unknown':<20} | PKL Only")

    print("---------------------------\n")

def delete_user(db):
    student_id = input("Enter Student ID to delete: ").strip()
    if not student_id: return

    # 1. Delete from DB and Get Section
    section = db.delete_student(student_id)
    
    if not section:
        print(f"Student ID '{student_id}' not found in database or could not be deleted.")
        return

    print(f"Student belonged to section: '{section}'")

    # 2. Delete from Specific Section File
    section_file = f"data/embeddings/{section}.pkl"
    if os.path.exists(section_file):
        try:
            with open(section_file, 'rb') as f: data = pickle.load(f)
            if student_id in data:
                del data[student_id]
                with open(section_file, 'wb') as f: pickle.dump(data, f)
                print(f"[System] Removed '{student_id}' from {section_file}.")
            else:
                print(f"[System] '{student_id}' not found in {section_file}.")
        except Exception as e:
            print(f"[Error] Failed to update section file: {e}")
    else:
        print(f"[System] Section file {section_file} not found.")

    # 3. Scrub from Persistent Caches
    cache_dir = "data/persistent_cache"
    if os.path.exists(cache_dir):
        scrub_count = 0
        for fname in os.listdir(cache_dir):
            if fname.startswith(f"cache_{section}_") and fname.endswith(".pkl"):
                fpath = os.path.join(cache_dir, fname)
                try:
                    with open(fpath, 'rb') as f: cache_data = pickle.load(f)
                    if student_id in cache_data:
                        del cache_data[student_id]
                        with open(fpath, 'wb') as f: pickle.dump(cache_data, f)
                        scrub_count += 1
                except: pass
        print(f"[System] Scrubbed '{student_id}' from {scrub_count} persistent cache files.")

    print(f"Successfully deleted '{student_id}' from all records.")

def list_sessions(db):
    print("\n--- Session Summary ---")
    try:
        rows = db.get_all_sessions_summary() # New method
    except AttributeError:
        print("Error: DBManager missing 'get_all_sessions_summary'")
        return

    if not rows:
        print("No sessions recorded.")
    else:
        # s.session_name, s.section, s.start_time, total, present
        print(f"{'Session':<20} | {'Section':<10} | {'Time':<20} | {'Attended':<10}")
        print("-" * 75)
        for name, section, start, total, present in rows:
            # attended_str = f"{present}/{total}"
            # Formatting timestamp
            try:
                t_obj = datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S.%f")
                t_str = t_obj.strftime("%m-%d %H:%M")
            except: t_str = str(start)[:16]
            
            print(f"{name:<20} | {section:<10} | {t_str:<20} | {present}/{total}")
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
