import sqlite3
from datetime import datetime, date

DB_NAME = "attendance.db"


def init_db(initial_admin: str = None):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT CHECK(status IN ('present', 'absent')) DEFAULT 'present',
                UNIQUE(student_name, date)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        if initial_admin:
            cursor.execute("""
                INSERT OR IGNORE INTO admins (username, created_at)
                VALUES (?, ?)
            """, (initial_admin.lower().strip(), datetime.now().isoformat()))
        conn.commit()


def is_user_admin(username: str) -> bool:
    if not username:
        return False
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM admins WHERE username = ?", (username.lower().strip(),))
        return cursor.fetchone() is not None


def add_admin(username: str):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO admins (username, created_at)
            VALUES (?, ?)
        """, (username.lower().strip(), datetime.now().isoformat()))
        conn.commit()


def get_all_admins():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, created_at FROM admins ORDER BY id ASC")
        return cursor.fetchall()


def remove_admin(username: str):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM admins WHERE username = ?", (username.lower().strip(),))
        conn.commit()


def mark_attendance(student_name: str, target_date: str = None, status: str = "present", target_time: str = None):
    if not target_date:
        target_date = date.today().isoformat()
    if not target_time:
        target_time = datetime.now().strftime("%H:%M:%S")

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance (student_name, date, time, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(student_name, date) DO UPDATE SET
                status = excluded.status,
                time = excluded.time
        """, (student_name, target_date, target_time, status))
        conn.commit()


def get_attendance_by_date(target_date: str, all_students: list):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_name, time, status FROM attendance WHERE date = ?", (target_date,))
        records = {row[0]: {"time": row[1], "status": row[2]} for row in cursor.fetchall()}

    result = []
    for name in all_students:
        if name in records:
            result.append({
                "name": name,
                "status": records[name]["status"],
                "time": records[name]["time"]
            })
        else:
            result.append({
                "name": name,
                "status": "absent",
                "time": "—"
            })
    return result

def get_attendance_for_period(start_date: str, end_date: str):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT student_name, date, time, status 
            FROM attendance 
            WHERE date BETWEEN ? AND ?
            ORDER BY date ASC, student_name ASC
        """, (start_date, end_date))
        return cursor.fetchall()