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
        
        # Phase 2c: Attendance Intervals for tracking entry/exit times
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_intervals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                student_id TEXT NOT NULL,
                student_name TEXT,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                duration_minutes REAL,
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
        
        # Create Session with explicit IST timestamp
        cursor.execute(
            "INSERT INTO sessions (session_name, section, start_time) VALUES (?, ?, ?)", 
            (session_name, section, datetime.now().isoformat())
        )
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
            now = datetime.now().isoformat()
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

    # ============================================
    # PHASE 2c: INTERVAL LOGGING METHODS
    # ============================================
    
    def start_interval(self, session_id, student_id, student_name):
        """
        Opens a new attendance interval when student becomes present.
        Returns the interval_id for later closing.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO attendance_intervals (session_id, student_id, student_name, entry_time)
                VALUES (?, ?, ?, ?)
            ''', (session_id, student_id, student_name, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"[DB] Start Interval Error: {e}")
            return None
        finally:
            conn.close()
    
    def close_interval(self, interval_id, exit_time=None):
        """
        Closes an open interval when student becomes absent.
        Calculates duration_minutes automatically.
        """
        if not interval_id:
            return
        if exit_time is None:
            exit_time = datetime.now().isoformat()
            
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Get entry_time to calculate duration
            cursor.execute('SELECT entry_time FROM attendance_intervals WHERE id = ?', (interval_id,))
            row = cursor.fetchone()
            if row and row[0]:
                entry_time = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                exit_time_dt = datetime.fromisoformat(exit_time) if isinstance(exit_time, str) else exit_time
                duration = (exit_time_dt - entry_time).total_seconds() / 60.0
            else:
                duration = 0.0
            
            cursor.execute('''
                UPDATE attendance_intervals 
                SET exit_time = ?, duration_minutes = ?
                WHERE id = ?
            ''', (exit_time, duration, interval_id))
            conn.commit()
        except Exception as e:
            print(f"[DB] Close Interval Error: {e}")
        finally:
            conn.close()
    
    def close_all_intervals(self, session_id, exit_time=None):
        """
        Session end safety: Closes all open intervals for a session.
        Called when session stops to ensure no dangling intervals.
        """
        if exit_time is None:
            exit_time = datetime.now().isoformat()
            
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Find all open intervals (exit_time is NULL)
            cursor.execute('''
                SELECT id, entry_time FROM attendance_intervals 
                WHERE session_id = ? AND exit_time IS NULL
            ''', (session_id,))
            open_intervals = cursor.fetchall()
            
            for interval_id, entry_time_str in open_intervals:
                if entry_time_str:
                    entry_time = datetime.fromisoformat(entry_time_str) if isinstance(entry_time_str, str) else entry_time_str
                    exit_time_dt = datetime.fromisoformat(exit_time) if isinstance(exit_time, str) else exit_time
                    duration = (exit_time_dt - entry_time).total_seconds() / 60.0
                else:
                    duration = 0.0
                
                cursor.execute('''
                    UPDATE attendance_intervals 
                    SET exit_time = ?, duration_minutes = ?
                    WHERE id = ?
                ''', (exit_time, duration, interval_id))
            
            conn.commit()
            print(f"[DB] Closed {len(open_intervals)} open intervals for session {session_id}")
        except Exception as e:
            print(f"[DB] Close All Intervals Error: {e}")
        finally:
            conn.close()
    
    def get_session_intervals(self, session_id):
        """
        Returns all intervals for a session (for reports).
        Returns list of dicts with interval data.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT id, student_id, student_name, entry_time, exit_time, duration_minutes
                FROM attendance_intervals
                WHERE session_id = ?
                ORDER BY student_id, entry_time
            ''', (session_id,))
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "student_id": r[1],
                    "student_name": r[2],
                    "entry_time": r[3],
                    "exit_time": r[4],
                    "duration_minutes": r[5] or 0.0
                }
                for r in rows
            ]
        except Exception as e:
            print(f"[DB] Get Session Intervals Error: {e}")
            return []
        finally:
            conn.close()

if __name__ == "__main__":
    db = DBManager()
    print("DB Initialized.")
