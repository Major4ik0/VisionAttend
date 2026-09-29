import sqlite3
from datetime import datetime, date

DB_NAME = "attendance.db"

def init_db():
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
