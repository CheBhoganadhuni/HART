import sqlite3
import time
from datetime import datetime, timedelta
import os

class DBManager:
    def __init__(self, db_path="data/attendance.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        """Returns a connection with WAL mode and timeout for concurrent access."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table: students
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                section TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ... (attendance_logs table creation - no change needed or deprecated) ...
        
        # Schema Migration: Check if student_id column exists
        try:
            cursor.execute('SELECT student_id FROM students LIMIT 1')
        except sqlite3.OperationalError:
            print("[DB] Adding missing column 'student_id' to students")
            cursor.execute('ALTER TABLE students ADD COLUMN student_id TEXT')
            
        # Schema Migration: Check if session_name column exists in session_attendance
        try:
            cursor.execute('SELECT session_name FROM session_attendance LIMIT 1')
        except sqlite3.OperationalError:
             print("[DB] Adding missing column 'session_name' to session_attendance")
             cursor.execute('ALTER TABLE session_attendance ADD COLUMN session_name TEXT')
        
        # New Tables for Modular Sections
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                section TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE', -- ACTIVE, CLOSED
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                session_name TEXT, -- Added for easier querying
                student_id TEXT NOT NULL,
                status TEXT DEFAULT 'Absent', -- Present, Absent
                last_seen TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS unknown_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                session_name TEXT,
                track_id INTEGER,
                image_path TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def register_student(self, name, section, student_id):
        """Registers a student with section and ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT OR REPLACE INTO students (name, section, student_id) VALUES (?, ?, ?)", (name, section, student_id))
            conn.commit()
            print(f"[DB] Student '{name}' ({student_id}) registered in section '{section}'.")
        except Exception as e:
            print(f"[DB] Error registering student: {e}")
        finally:
            conn.close()
            
    def delete_student(self, student_id):
        """Deletes a student by ID and returns their section for file cleanup."""
        conn = self.get_connection()
        cursor = conn.cursor()
        section = None
        try:
            cursor.execute("SELECT section FROM students WHERE student_id = ?", (student_id,))
            row = cursor.fetchone()
            if row:
                section = row[0]
                cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
                conn.commit()
                print(f"[DB] Student {student_id} deleted from database.")
                return section
            print(f"[DB] Student {student_id} not found.")
            return None
        except Exception as e:
            print(f"[DB] Error deleting student: {e}")
            return None
        finally:
            conn.close()
            
    def delete_section_data(self, section):
        """Deletes all students in a section."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM students WHERE section = ?", (section,))
            conn.commit()
            print(f"[DB] All students in section '{section}' deleted.")
        except Exception as e:
            print(f"[DB] Error deleting section data: {e}")
        finally:
            conn.close()
            
    def get_section_students(self, section):
        """Returns dict of {student_id: name} in a section."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name FROM students WHERE section = ?", (section,))
        rows = cursor.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def start_session(self, session_name, section):
        """Starts a new session and initializes attendance."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create Session
        cursor.execute("INSERT INTO sessions (session_name, section) VALUES (?, ?)", (session_name, section))
        session_id = cursor.lastrowid
        
        # Initialize Attendance for all section students (By ID)
        students = self.get_section_students(section) # Returns dict {id: name}
        if students:
            data = [(session_id, session_name, sid, 'Absent') for sid in students.keys()]
            cursor.executemany("INSERT INTO session_attendance (session_id, session_name, student_id, status) VALUES (?, ?, ?, ?)", data)
            
        conn.commit()
        conn.close()
        return session_id
        
    def close_session(self, session_id):
        """Closes a session and records end time."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            now = datetime.now()
            cursor.execute("UPDATE sessions SET status = 'CLOSED', end_time = ? WHERE id = ?", (now, session_id))
            conn.commit()
            print(f"[DB] Session {session_id} closed at {now}.")
        except Exception as e:
             print(f"[DB] Error closing session: {e}")
        finally:
            conn.close()

    def global_heartbeat_sync(self, session_id, present_ids):
        """Atomic update: Reset session to Absent, then mark present_ids as Present."""
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        
        try:
            # 1. Atomic Reset: Mark EVERYONE in this session as 'Absent'
            cursor.execute("UPDATE session_attendance SET status = 'Absent' WHERE session_id = ?", (session_id,))

            # 2. Mark Present: Only those who are currently detected
            if present_ids:
                placeholders = ','.join(['?'] * len(present_ids))
                cursor.execute(f'''
                    UPDATE session_attendance 
                    SET status = 'Present', last_seen = ? 
                    WHERE session_id = ? AND student_id IN ({placeholders})
                ''', (now, session_id, *present_ids))
            
            conn.commit()
        except Exception as e:
            print(f"[DB] Heartbeat Sync Error: {e}")
        finally:
            conn.close()

    def log_unknown_detection(self, session_id, session_name, track_id, image_path):
        """Logs an unknown detection event."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Use local timestamp instead of SQLite's UTC default
            local_timestamp = datetime.now().isoformat()
            cursor.execute("INSERT INTO unknown_detections (session_id, session_name, track_id, image_path, timestamp) VALUES (?, ?, ?, ?, ?)", 
                           (session_id, session_name, track_id, image_path, local_timestamp))
            conn.commit()
        except Exception as e:
            print(f"[DB] Error logging unknown detection: {e}")
        finally:
            conn.close()



    def get_student_by_id(self, student_id):
        """Checks if a student ID exists."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    # Management Helpers
    def get_all_students_with_section(self):
        """Returns list of (name, section, student_id) for all students."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, section, student_id FROM students ORDER BY section, name")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_all_sessions_summary(self):
        """Returns summary of all sessions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.session_name, s.section, s.start_time, COUNT(sa.id) as total, 
                   SUM(CASE WHEN sa.status='Present' THEN 1 ELSE 0 END) as present
            FROM sessions s
            LEFT JOIN session_attendance sa ON s.id = sa.session_id
            GROUP BY s.id
            ORDER BY s.start_time DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        return rows

if __name__ == "__main__":
    db = DBManager()
    print("DB Initialized.")
