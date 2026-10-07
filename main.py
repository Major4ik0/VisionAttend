import os
import json
import time
import shutil
from contextlib import asynccontextmanager
from datetime import date
from ldap3 import Server, Connection, ALL, SIMPLE
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from database import get_attendance_for_period
from excel_export import generate_attendance_report

from database import (
    init_db, mark_attendance, get_attendance_by_date,
    is_user_admin, add_admin, get_all_admins, remove_admin
)
from camera import FaceRecognitionCamera

CONFIG_FILE = "config.json"
UPLOADS_DIR = "uploads"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "video_source": "0",
        "ad_server": "192.168.1.10",
        "ad_domain": "CORP.LOCAL",
        "initial_admin": "admin"
    }


def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


config = load_config()
os.makedirs(UPLOADS_DIR, exist_ok=True)

cam = FaceRecognitionCamera(
    path_images=UPLOADS_DIR,
    video_source=config.get("video_source", "0")
)


def authenticate_ad(username: str, password: str) -> bool:
    # 1. Локальный режим разработки (без доступа к сети и LDAP)
    if config.get("dev_mode", False):
        dev_pass = config.get("dev_password", "admin")
        initial_admin = config.get("initial_admin", "admin")
        if username.lower() == initial_admin.lower() and password == dev_pass:
            print(f"🔧 Вход в DEV-режиме: {username}")
            return True
        return False

    server_ip = config.get("ad_server", "127.0.0.1")
    domain = config.get("ad_domain", "academy.org")

    # 2. Автоматическое формирование имени пользователя
    # Если подключение идет к локальному тестовому контейнеру OpenLDAP:
    if server_ip in ("127.0.0.1", "localhost", "host.docker.internal"):
        dc_suffix = ",".join([f"dc={part}" for part in domain.split(".")])
        if username.startswith("cn="):
            user_principal = username
        else:
            user_principal = f"cn={username},{dc_suffix}"
    else:
        # Для боевого Windows Server 2012 в академии (формат Active Directory UPN)
        if "@" in username or "\\" in username:
            user_principal = username
        else:
            user_principal = f"{username}@{domain}"

    try:
        print(f"[AUTH] Попытка входа: {user_principal} на сервер {server_ip}")
        server = Server(server_ip, port=389, get_info=ALL, connect_timeout=3)
        conn = Connection(server, user=user_principal, password=password, authentication=SIMPLE)
        result = conn.bind()
        if result:
            conn.unbind()
        return result
    except Exception as e:
        print(f"❌ Ошибка подключения к AD/LDAP: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(initial_admin=config.get("initial_admin"))
    cam.start_threads()
    yield
    cam.stop_threads()


app = FastAPI(lifespan=lifespan)
# Секретный ключ для подписи Cookie сессий
app.add_middleware(SessionMiddleware, secret_key="attend-secret-key-change-it")
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_all_student_names():
    names = set()
    if not os.path.exists(UPLOADS_DIR):
        return []
    for item in os.listdir(UPLOADS_DIR):
        item_path = os.path.join(UPLOADS_DIR, item)
        if os.path.isdir(item_path):
            names.add(item)
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            names.add(os.path.splitext(item)[0])
    return sorted(list(names))


def generate_camera_stream():
    while True:
        frame = cam.get_jpeg_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None}
    )


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    if authenticate_ad(username, password):
        request.session["user"] = username
        request.session["is_admin"] = is_user_admin(username)
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": "Неверный логин или пароль домена Windows"},
        status_code=401
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/video_feed")
def video_feed(request: Request):
    if not request.session.get("user"):
        raise HTTPException(status_code=401, detail="Необходима авторизация")
    return StreamingResponse(
        generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def view_attendance(request: Request, selected_date: str = None):
    current_user = request.session.get("user")
    if not current_user:
        return RedirectResponse(url="/login")

    is_admin = is_user_admin(current_user)
    request.session["is_admin"] = is_admin

    if not selected_date:
        selected_date = date.today().isoformat()

    students = get_all_student_names()
    records = get_attendance_by_date(selected_date, students)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "records": records,
            "selected_date": selected_date,
            "current_video_source": str(cam.video_source),
            "username": current_user,
            "is_admin": is_admin
        }
    )


@app.post("/update-camera")
def update_camera(request: Request, video_source: str = Form(...)):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    video_source = video_source.strip()
    config["video_source"] = video_source
    save_config(config)

    cam.change_source(video_source)
    return RedirectResponse(url="/", status_code=303)


@app.post("/update-attendance")
def update_attendance(request: Request,
                      student_name: str = Form(...),
                      selected_date: str = Form(...),
                      status: str = Form(...),
                      reason: str = Form(None)):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    mark_attendance(
        student_name=student_name,
        target_date=selected_date,
        status=status,
        reason=reason
    )
    return RedirectResponse(url=f"/?selected_date={selected_date}", status_code=303)


@app.get("/admins", response_class=HTMLResponse)
def manage_admins_page(request: Request):
    if not request.session.get("is_admin"):
        return RedirectResponse(url="/", status_code=303)

    admins = get_all_admins()
    return templates.TemplateResponse(
        request=request,
        name="admins.html",
        context={
            "admins": admins,
            "username": request.session.get("user")
        }
    )


@app.post("/admins/add")
def add_new_admin(request: Request, username: str = Form(...)):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    add_admin(username)
    return RedirectResponse(url="/admins", status_code=303)


@app.post("/admins/remove")
def delete_admin(request: Request, username: str = Form(...)):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    # Предотвращение случайного удаления себя
    if username.lower() != request.session.get("user", "").lower():
        remove_admin(username)

    return RedirectResponse(url="/admins", status_code=303)


@app.get("/export-attendance")
def export_attendance(request: Request, start_date: str = None, end_date: str = None):
    # Доступ разрешен только администраторам
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    today_str = date.today().isoformat()
    if not start_date:
        start_date = today_str
    if not end_date:
        end_date = today_str

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    students = get_all_student_names()
    records = get_attendance_for_period(start_date, end_date)

    file_stream = generate_attendance_report(start_date, end_date, students, records)
    filename = f"attendance_{start_date}_to_{end_date}.xlsx"

    return StreamingResponse(
        file_stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.post("/upload")
async def upload_student(request: Request, name: str = Form(...), file: UploadFile = File(...)):
    """Загрузка фотографии студента (поддерживает несколько фото в папку студента)."""
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    name = name.strip()
    student_dir = os.path.join(UPLOADS_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    # Сохраняем файл с уникальным именем на базе таймстемпа
    ext = os.path.splitext(file.filename)[1]
    filename = f"{int(time.time() * 1000)}{ext}"
    file_path = os.path.join(student_dir, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Перезагружаем векторную базу лиц в объекте камеры
    if hasattr(cam, "load_known_faces"):
        cam.load_known_faces()
    elif hasattr(cam, "load_fresh_faces"):
        cam.load_fresh_faces()

    return RedirectResponse(url="/upload", status_code=303)


@app.post("/capture-student")
def capture_student(request: Request, name: str = Form(...)):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    name = name.strip()
    if not name:
        return RedirectResponse(url="/upload", status_code=303)

    student_dir = os.path.join(UPLOADS_DIR, name)
    os.makedirs(student_dir, exist_ok=True)

    filename = f"{int(time.time() * 1000)}.jpg"
    file_path = os.path.join(student_dir, filename)

    # 1. Снимаем кадр из оперативной памяти
    if cam.save_snapshot(file_path):
        # 2. Добавляем только новый вектор под lock (занимает 0.1 сек, видеопоток не рвется)
        cam.add_single_face(file_path, name)
    else:
        raise HTTPException(status_code=500, detail="Камера ещё не успела получить кадр")

    return RedirectResponse(url="/upload", status_code=303)


@app.post("/delete-student")
def delete_student(request: Request, name: str = Form(...)):
    """Удаление студента (папки со всеми ракурсами или одиночного файла)."""
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    name = name.strip()
    student_dir = os.path.join(UPLOADS_DIR, name)

    # 1. Удаляем папку со всеми ракурсами
    if os.path.exists(student_dir) and os.path.isdir(student_dir):
        shutil.rmtree(student_dir)

    # 2. Проверяем одиночные файлы старого формата в корне uploads
    for filename in os.listdir(UPLOADS_DIR):
        base_name, _ = os.path.splitext(filename)
        if base_name == name:
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    if hasattr(cam, "load_known_faces"):
        cam.load_known_faces()
    elif hasattr(cam, "load_fresh_faces"):
        cam.load_fresh_faces()

    return RedirectResponse(url="/upload", status_code=303)


def get_all_students_with_photos():
    """Возвращает список студентов со списком ссылок на их фотографии."""
    students = {}
    if not os.path.exists(UPLOADS_DIR):
        return []

    for item in sorted(os.listdir(UPLOADS_DIR)):
        item_path = os.path.join(UPLOADS_DIR, item)
        if item.startswith('.'):
            continue

        # Папка студента с несколькими ракурсами
        if os.path.isdir(item_path):
            photos = [
                f"/uploads/{item}/{f}"
                for f in sorted(os.listdir(item_path))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            students[item] = photos

        # Одиночные фотографии старого формата
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(item)[0]
            if name not in students:
                students[name] = []
            students[name].append(f"/uploads/{item}")

    return [{"name": name, "photos": photos} for name, photos in students.items()]


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    if not request.session.get("is_admin"):
        return RedirectResponse(url="/", status_code=303)

    students = get_all_students_with_photos()
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"students": students, "username": request.session.get("user")}
    )


@app.post("/delete-photo")
def delete_photo(request: Request, photo_path: str = Form(...)):
    """Удаление одной конкретной фотографии студента."""
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    # Предотвращение Path Traversal: извлекаем относительный путь внутри uploads
    clean_path = photo_path.replace("/uploads/", "").lstrip("/\\")
    full_path = os.path.abspath(os.path.join(UPLOADS_DIR, clean_path))

    # Удаляем файл только если он строго внутри UPLOADS_DIR
    if full_path.startswith(os.path.abspath(UPLOADS_DIR)) and os.path.isfile(full_path):
        os.remove(full_path)

    # Обновляем базу дескрипторов лиц камеры
    if hasattr(cam, "load_known_faces"):
        cam.load_known_faces()
    elif hasattr(cam, "load_fresh_faces"):
        cam.load_fresh_faces()

    return RedirectResponse(url="/upload", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


