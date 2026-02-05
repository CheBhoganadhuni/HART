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
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table: attendance_logs
        # session_id can be a UUID or just auto-increment. 
        # We will use auto-increment ID as the session handle mostly, 
        # but let's keep it simple.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                session_start TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                duration_seconds REAL DEFAULT 0
            )
        ''')
        
        # Schema Migration: Check if session_name column exists
        try:
            cursor.execute('SELECT session_name FROM attendance_logs LIMIT 1')
        except sqlite3.OperationalError:
            print("[DB] Adding missing column 'session_name' to attendance_logs")
            cursor.execute('ALTER TABLE attendance_logs ADD COLUMN session_name TEXT DEFAULT "Default Session"')
        
        conn.commit()
        conn.close()

    def register_student(self, name):
        """Registers a student if not already in DB."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
            conn.commit()
            print(f"[DB] Student '{name}' registered/verified in database.")
        except Exception as e:
            print(f"[DB] Error registering student: {e}")
        finally:
            conn.close()
            
    def delete_student(self, name):
        """Deletes a student from the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM students WHERE name = ?", (name,))
            # Optional: Delete logs too? Usually better to keep logs for history, 
            # but user said "Delete a specific student", implies full cleanup.
            # Let's keep logs for now unless requested, or maybe delete for privacy.
            # I'll stick to deleting the student registration.
            changes = cursor.rowcount
            
            # Check if table is empty and reset sequence
            cursor.execute("SELECT COUNT(*) FROM students")
            count = cursor.fetchone()[0]
            if count == 0:
                print("[DB] Table 'students' is empty. Resetting ID sequence.")
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='students'")
            
            conn.commit()
            print(f"[DB] Student '{name}' deleted from database. Rows affected: {changes}")
            return changes > 0
        except Exception as e:
            print(f"[DB] Error deleting student: {e}")
            return False
        finally:
            conn.close()
            
    def get_all_students(self):
        """Returns a list of all registered student names."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM students")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def log_heartbeat(self, student_name, session_name="Class Session"):
        """
        Updates the last_seen timestamp for the student.
        Logic:
        - Find the most recent session for this student.
        - If last_seen is within 60 seconds of now, update last_seen and duration.
        - If > 60 seconds or no session exists, create a new session.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        
        # Check for the latest session for this student
        cursor.execute('''
            SELECT id, last_seen, session_start FROM attendance_logs 
            WHERE student_name = ? 
            ORDER BY last_seen DESC 
            LIMIT 1
        ''', (student_name,))
        
        row = cursor.fetchone()
        
        status = "UNKNOWN"
        
        if row:
            session_id, last_seen_str, session_start_str = row
            try:
                last_seen = datetime.fromisoformat(last_seen_str) if 'T' in last_seen_str else datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S.%f")
                session_start = datetime.fromisoformat(session_start_str) if 'T' in session_start_str else datetime.strptime(session_start_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                 last_seen = now
                 session_start = now

            time_diff = (now - last_seen).total_seconds()
            
            if time_diff <= 30:
                # Update existing session
                new_duration = (now - session_start).total_seconds()
                cursor.execute('''
                    UPDATE attendance_logs 
                    SET last_seen = ?, duration_seconds = ? 
                    WHERE id = ?
                ''', (now, new_duration, session_id))
                status = "UPDATED"
            else:
                # Gap too large, start new session
                cursor.execute('''
                    INSERT INTO attendance_logs (student_name, session_start, last_seen, duration_seconds, session_name)
                    VALUES (?, ?, ?, 0, ?)
                ''', (student_name, now, now, session_name))
                status = "NEW_SESSION"
        else:
            # No previous record, start first session
            cursor.execute('''
                INSERT INTO attendance_logs (student_name, session_start, last_seen, duration_seconds, session_name)
                VALUES (?, ?, ?, 0, ?)
            ''', (student_name, now, now, session_name))
            status = "NEW_SESSION"
            
        conn.commit()
        conn.close()
        return status

    def get_sessions_summary(self):
        """Returns a summary of sessions: Name, Count of Unique Students, Earliest Time."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Aggregate by session_name
        cursor.execute('''
            SELECT session_name, COUNT(DISTINCT student_name), MIN(session_start)
            FROM attendance_logs
            GROUP BY session_name
            ORDER BY MIN(session_start) DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_total_minutes(self, student_name):
        """Calculates total minutes attended across all sessions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT SUM(duration_seconds) FROM attendance_logs 
            WHERE student_name = ?
        ''', (student_name,))
        
        result = cursor.fetchone()[0]
        conn.close()
        
        if result:
            return round(result / 60, 2)
        return 0.0

# Quick test if run directly
if __name__ == "__main__":
    db = DBManager()
    print("DB Initialized.")
