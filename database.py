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
        try:
            cursor.execute("ALTER TABLE attendance ADD COLUMN reason TEXT DEFAULT ''")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Колонка уже существует
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


def mark_attendance(student_name: str, target_date: str = None, status: str = 'present', reason: str = None):
    """Отметка или обновление статуса и причины отсутствия."""
    from datetime import date, datetime
    if not target_date:
        target_date = date.today().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, reason FROM attendance WHERE student_name = ? AND date = ?",
            (student_name, target_date)
        )
        row = cursor.fetchone()
        # Если передана новая причина — используем её, иначе оставляем старую
        new_reason = reason if reason is not None else (row[1] if row and row[1] else "")
        if row:
            cursor.execute(
                """
                UPDATE attendance 
                SET status = ?, time = ?, reason = ?
                WHERE id = ?
                """,
                (status, current_time, new_reason, row[0])
            )
        else:
            cursor.execute(
                """
                INSERT INTO attendance (student_name, date, time, status, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_name, target_date, current_time, status, new_reason)
            )
        conn.commit()


def get_attendance_by_date(target_date: str, all_students: list):
    records_dict = {}
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_name, time, status, reason FROM attendance WHERE date = ?",
            (target_date,)
        )
        for row in cursor.fetchall():
            records_dict[row[0]] = {
                "time": row[1],
                "status": row[2],
                "reason": row[3] or ""
            }

    result = []
    for student in all_students:
        if student in records_dict:
            rec = records_dict[student]
            result.append({
                "name": student,
                "status": rec["status"],
                "time": rec["time"],
                "reason": rec["reason"]
            })
        else:
            result.append({
                "name": student,
                "status": "absent",
                "time": "—",
                "reason": ""
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