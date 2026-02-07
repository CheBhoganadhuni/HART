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
        return sqlite3.connect(self.db_path)

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
            # Note: Existing rows will have NULL student_id. User might need to clear DB or we handle it.
            # For now, let's assume fresh start or user clears old data if conflict.
        
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
                student_id TEXT NOT NULL,
                status TEXT DEFAULT 'Absent', -- Present, Absent
                last_seen TIMESTAMP,
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
            data = [(session_id, sid, 'Absent') for sid in students.keys()]
            cursor.executemany("INSERT INTO session_attendance (session_id, student_id, status) VALUES (?, ?, ?)", data)
            
        conn.commit()
        conn.close()
        return session_id

    def global_heartbeat_sync(self, session_id, present_ids):
        """Atomic update: Mark present_ids as PRESENT, others as ABSENT."""
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        
        try:
            if present_ids:
                # Mark Present
                placeholders = ','.join(['?'] * len(present_ids))
                cursor.execute(f'''
                    UPDATE session_attendance 
                    SET status = 'Present', last_seen = ? 
                    WHERE session_id = ? AND student_id IN ({placeholders})
                ''', (now, session_id, *present_ids))
                
                # Mark Absent (Everyone else in this session)
                cursor.execute(f'''
                    UPDATE session_attendance 
                    SET status = 'Absent' 
                    WHERE session_id = ? AND student_id NOT IN ({placeholders})
                ''', (session_id, *present_ids))
            else:
                 # Mark ALL Absent if no one is present
                 cursor.execute("UPDATE session_attendance SET status = 'Absent' WHERE session_id = ?", (session_id,))

            conn.commit()
        except Exception as e:
            print(f"[DB] Heartbeat Sync Error: {e}")
        finally:
            conn.close()

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
