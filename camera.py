import os
import time
import cv2
import numpy as np
import face_recognition
import threading
import pickle
from queue import Queue
from collections import deque, Counter


class FaceRecognitionCamera:
    def __init__(self, path_images: str, video_source=0, delay=15.0):
        self.path_images = path_images
        self.video_source = self._parse_source(video_source)
        self.delay = delay
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_detected_name = ""
        self.last_detection_time = 0
        self.cache_file = os.path.join(path_images, "face_cache.pkl")

        # КРИТИЧНО: блокировка для всех вызовов dlib
        self.dlib_lock = threading.Lock()

        # Очереди
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)

        # Флаги потоков
        self.running = False
        self.processing = False

        # Сглаживание имени
        self.name_history = deque(maxlen=5)
        self.display_name = "Unknown"

        # FPS
        self.capture_fps = 0
        self.process_fps = 0

        self.cap = None
        self.output_frame = None
        self.frame_lock = threading.Lock()

        self.load_known_faces()

    def _parse_source(self, source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    # ---------- Загрузка лиц ----------

    def load_known_faces(self):
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images, exist_ok=True)
            return

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                self.known_face_encodings = cache['encodings']
                self.known_face_names = cache['names']

                # Проверяем, не изменились ли файлы
                current = {f for f in os.listdir(self.path_images)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
                cached = set(cache.get('files', []))
                if current != cached:
                    print("🔄 Файлы изменились, перезагружаем...")
                    self.load_fresh_faces()
                else:
                    print(f"✅ Загружено из кэша: {len(self.known_face_names)} лиц")
                return
            except Exception as e:
                print(f"❌ Ошибка кэша: {e}, грузим заново")
                self.load_fresh_faces()
        else:
            self.load_fresh_faces()

    def load_fresh_faces(self):
        """Загрузка лиц. dlib вызывается под lock, чтобы не конфликтовать с process_frames."""
        print("=== ЗАГРУЗКА ЛИЦ ===")
        new_encodings = []
        new_names = []
        files_list = []

        # Сначала готовим список файлов
        image_files = [f for f in os.listdir(self.path_images)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # ВАЖНО: сам расчёт энкодингов — под lock
        with self.dlib_lock:
            for filename in image_files:
                image_path = os.path.join(self.path_images, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        new_encodings.append(encodings[0])
                        new_names.append(os.path.splitext(filename)[0])
                        files_list.append(filename)
                        print(f"  ✅ {filename}")
                    else:
                        print(f"  ❌ Лицо не найдено: {filename}")
                except Exception as e:
                    print(f"  ❌ Ошибка {filename}: {e}")

        # Под тем же lock заменяем списки
        with self.dlib_lock:
            self.known_face_encodings = new_encodings
            self.known_face_names = new_names

        print(f"✅ Всего загружено: {len(new_names)} лиц")

        if new_encodings:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({
                        'encodings': new_encodings,
                        'names': new_names,
                        'files': files_list,
                        'timestamp': time.time()
                    }, f)
                print(f"✅ Кэш сохранён: {self.cache_file}")
            except Exception as e:
                print(f"❌ Ошибка сохранения кэша: {e}")

    # ---------- Поток захвата ----------

    def capture_frames(self):
        print("📸 Поток захвата запущен")
        frame_count = 0
        fps_start = time.time()

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.05)
                continue

            # Отражение только для веб-камер
            if isinstance(self.video_source, int):
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (640, 380))

            frame_count += 1
            if frame_count >= 30:
                self.capture_fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except:
                    pass

    # ---------- Поток обработки ----------

    def process_frames(self):
        print("⚙️ Поток обработки запущен")
        frame_count = 0
        fps_start = time.time()

        while self.processing:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.001)
                    continue

                frame = self.frame_queue.get()

                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # ВАЖНО: все вызовы dlib — под lock
                with self.dlib_lock:
                    face_locations = face_recognition.face_locations(rgb)
                    face_encodings = face_recognition.face_encodings(rgb, face_locations)

                    # Сравнение тоже под lock (compare_faces тоже использует dlib/numpy)
                    current_name = "Unknown"
                    matched_index = None
                    if face_encodings and self.known_face_encodings:
                        closest = int(np.argmin([top for (top, _, _, _) in face_locations]))
                        enc = face_encodings[closest]
                        matches = face_recognition.compare_faces(self.known_face_encodings, enc)
                        if True in matches:
                            matched_index = matches.index(True)
                            current_name = self.known_face_names[matched_index]

                # Масштабируем координаты обратно
                face_locations = [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in face_locations]

                # Запись отметки
                if current_name != "Unknown":
                    now = time.time()
                    if (current_name != self.last_detected_name or
                            now - self.last_detection_time > self.delay):
                        self.save_attendance(current_name)
                        self.last_detected_name = current_name
                        self.last_detection_time = now

                # Сглаживание
                self.name_history.append(current_name)
                if self.name_history:
                    self.display_name = Counter(self.name_history).most_common(1)[0][0]

                # Рисуем на кадре
                display = frame.copy()
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    is_closest = (matched_index is not None and
                                  i == int(np.argmin([t for (t, _, _, _) in face_locations])))
                    color = (0, 255, 0) if (is_closest and self.display_name != "Unknown") else (0, 165, 255)
                    label = self.display_name if is_closest else "Unknown"

                    cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display, label, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                # Инфо-строка
                cv2.putText(display, f"Cap: {self.capture_fps:.1f} FPS | Proc: {self.process_fps:.1f} FPS",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                with self.frame_lock:
                    self.output_frame = display

                frame_count += 1
                if frame_count >= 30:
                    self.process_fps = frame_count / (time.time() - fps_start)
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                time.sleep(0.05)

    # ---------- Отдача кадра в HTTP ----------

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.output_frame is None:
                return None
            ok, jpeg = cv2.imencode('.jpg', self.output_frame)
            if not ok:
                return None
            return jpeg.tobytes()

    # ---------- Управление источником ----------

    def change_source(self, new_source):
        self.stop_threads()
        if self.cap and self.cap.isOpened():
            self.cap.release()

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        self.video_source = self._parse_source(new_source)
        self.start_threads()

    def start_threads(self):
        self.cap = cv2.VideoCapture(self.video_source)

        if isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'http://')):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.processing = True

        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()

    def stop_threads(self):
        self.running = False
        self.processing = False
        time.sleep(0.3)

    # ---------- Запись посещаемости ----------

    def save_attendance(self, name):
        # Импорт внутри метода, чтобы не было циклической зависимости
        from database import mark_attendance
        try:
            mark_attendance(student_name=name)
            print(f"✅ Отмечен: {name}")
        except Exception as e:
            print(f"❌ Ошибка отметки: {e}")