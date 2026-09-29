import os
import json
import time
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from database import init_db, mark_attendance, get_attendance_by_date
from camera import FaceRecognitionCamera

CONFIG_FILE = "config.json"
UPLOADS_DIR = "uploads"


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"video_source": "0"}


def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


config = load_config()
os.makedirs(UPLOADS_DIR, exist_ok=True)

cam = FaceRecognitionCamera(
    path_images=UPLOADS_DIR,
    video_source=config.get("video_source", "0")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cam.start_threads()
    yield
    cam.stop_threads()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


def get_all_student_names():
    return [os.path.splitext(f)[0]
            for f in os.listdir(UPLOADS_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


def generate_camera_stream():
    while True:
        frame = cam.get_jpeg_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def view_attendance(request: Request, selected_date: str = None):
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
            "current_video_source": str(cam.video_source)
        }
    )


@app.post("/update-camera")
def update_camera(video_source: str = Form(...)):
    video_source = video_source.strip()
    config["video_source"] = video_source
    save_config(config)

    cam.change_source(video_source)
    return RedirectResponse(url="/", status_code=303)


@app.post("/update-attendance")
def update_attendance(student_name: str = Form(...),
                      selected_date: str = Form(...),
                      status: str = Form(...)):
    mark_attendance(student_name=student_name,
                    target_date=selected_date,
                    status=status)
    return RedirectResponse(url=f"/?selected_date={selected_date}", status_code=303)


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    students = get_all_student_names()
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"students": students}
    )


@app.post("/upload")
async def upload_student(name: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    filename = f"{name}{ext}"
    file_path = os.path.join(UPLOADS_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # КРИТИЧНО: этот вызов трогает dlib.
    # Внутри camera.py он защищён dlib_lock, поэтому безопасен.
    cam.load_fresh_faces()

    return RedirectResponse(url="/upload", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)