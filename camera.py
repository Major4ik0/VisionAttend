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
        # Блокировка для всех вызовов dlib во избежание Segmentation Fault
        self.dlib_lock = threading.Lock()
        # Очереди кадров между потоками
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        # Флаги управления потоками
        self.running = False
        self.processing = False
        # Сглаживание предсказаний имени (Temporal Smoothing)
        self.name_history = deque(maxlen=5)
        self.display_name = "Unknown"
        # Метрики FPS
        self.capture_fps = 0
        self.process_fps = 0
        self.cap = None
        self.output_frame = None  # Кадр с нарисованными рамками и текстом для стриминга
        self.raw_frame = None     # Чистый исходный кадр для снимков студентов
        self.frame_lock = threading.Lock()
        self.load_known_faces()

    @staticmethod
    def _parse_source(source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    # ---------- Загрузка и кэширование лиц ----------

    def load_known_faces(self):
        """Загрузка лиц: поддерживает как одиночные файлы, так и папки студентов."""
        if not os.path.exists(self.path_images):
            return

        self.known_face_encodings = []
        self.known_face_names = []

        for item in os.listdir(self.path_images):
            item_path = os.path.join(self.path_images, item)

            # 1. Если это папка студента (uploads/Иванов И.И./)
            if os.path.isdir(item_path):
                student_name = item
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(item_path, filename)
                        self._add_face_encoding(img_path, student_name)

            # 2. Одиночные старые файлы в корне uploads/
            elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                student_name = os.path.splitext(item)[0]
                self._add_face_encoding(item_path, student_name)

        print(f"Всего загружено векторов лиц: {len(self.known_face_encodings)} для людей: {set(self.known_face_names)}")

    def _add_face_encoding(self, image_path: str, name: str):
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
        except Exception as e:
            print(f"Ошибка чтения {image_path}: {e}")

    def load_fresh_faces(self):
        """Загрузка изображений и расчет энкодингов под dlib_lock."""
        print("=== ЗАГРУЗКА ЛИЦ ===")
        new_encodings = []
        new_names = []
        files_list = []

        image_files = [f for f in os.listdir(self.path_images)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

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
                        print(f"  ✅ Загружено: {filename}")
                    else:
                        print(f"  ❌ Лицо не найдено на снимке: {filename}")
                except Exception as e:
                    print(f"  ❌ Ошибка обработки {filename}: {e}")

        with self.dlib_lock:
            self.known_face_encodings = new_encodings
            self.known_face_names = new_names

        print(f"✅ Итого загружено: {len(new_names)} лиц")

        if new_encodings:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump({
                        'encodings': new_encodings,
                        'names': new_names,
                        'files': files_list,
                        'timestamp': time.time()
                    }, f)
                print(f"✅ Кэш обновлен: {self.cache_file}")
            except Exception as e:
                print(f"❌ Ошибка записи кэша: {e}")

    # ---------- Поток видеозахвата ----------

    def capture_frames(self):
        """Безопасный захват кадров без переполнения очереди."""
        print("📹 Поток захвата запущен")
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Защита от переполнения: если очередь полна, удаляем старый кадр
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    # ---------- Поток распознавания и детекции ----------

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

                # Уменьшаем в 2 раза для ускорения работы детектора
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                # Вызовы face_recognition и dlib строго под lock
                with self.dlib_lock:
                    face_locations = face_recognition.face_locations(rgb)
                    face_encodings = face_recognition.face_encodings(rgb, face_locations)

                    current_name = "Unknown"
                    matched_index = None

                    if face_encodings and self.known_face_encodings:
                        # 1. Находим ближайшее/основное лицо (с наименьшей координатой top)
                        closest_face_index = int(np.argmin([top for (top, right, bottom, left) in face_locations]))
                        closest_face_encoding = face_encodings[closest_face_index]
                        matched_index = closest_face_index

                        # 2. Сравниваем его вектор со всеми сохраненными ракурсами
                        face_distances = face_recognition.face_distance(self.known_face_encodings,
                                                                        closest_face_encoding)
                        best_match_index = np.argmin(face_distances)

                        # 3. Порог 0.5 гарантирует минимум ложных срабатываний (по умолчанию в dlib 0.6)
                        if face_distances[best_match_index] < 0.5:
                            current_name = self.known_face_names[best_match_index]

                # Масштабируем координаты рамок обратно к исходному размеру
                face_locations = [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in face_locations]

                # Запись посещения в БД при фиксации лица
                if current_name != "Unknown":
                    now = time.time()
                    if (current_name != self.last_detected_name or
                            now - self.last_detection_time > self.delay):
                        self.save_attendance(current_name)
                        self.last_detected_name = current_name
                        self.last_detection_time = now

                # Сглаживание предсказаний
                self.name_history.append(current_name)
                if self.name_history:
                    self.display_name = Counter(self.name_history).most_common(1)[0][0]

                # Отрисовка рамок и подписей для веб-стрима
                display = frame.copy()
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    is_closest = (matched_index is not None and i == matched_index)
                    color = (0, 255, 0) if (is_closest and self.display_name != "Unknown") else (0, 165, 255)
                    label = self.display_name if is_closest else "Unknown"

                    cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display, label, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

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
                print(f"❌ Ошибка в потоке обработки: {e}")
                time.sleep(0.05)

    # ---------- Отдача кадра и снимков ----------

    def get_jpeg_frame(self):
        """Возвращает сжатый кадр с рамками для HTTP MJPEG стрима."""
        with self.frame_lock:
            if self.output_frame is None:
                return None
            ok, jpeg = cv2.imencode('.jpg', self.output_frame)
            if not ok:
                return None
            return jpeg.tobytes()

    def save_snapshot(self, file_path: str) -> bool:
        """Сохранение кадра из памяти без повторного обращения к self.cap."""
        try:
            with self.frame_lock:
                if self.output_frame is None:
                    return False
                frame_to_save = self.output_frame.copy()

            # Сохраняем изображение на диск
            return cv2.imwrite(file_path, frame_to_save)
        except Exception as e:
            print(f"❌ Ошибка сохранения снимка: {e}")
            return False

    def add_single_face(self, image_path: str, name: str):
        """Быстрое добавление одного лица без пересчета всей базы (без зависания)."""
        try:
            image = face_recognition.load_image_file(image_path)
            # Вызов dlib строго под тем же lock, что и в process_frames!
            with self.dlib_lock:
                encodings = face_recognition.face_encodings(image)

            if encodings:
                # Атомарное добавление в списки
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
                print(f"✅ Успешно добавлен ракурс для: {name}")
                return True
            else:
                print(f"⚠️ Лицо не обнаружено на снимке: {image_path}")
                return False
        except Exception as e:
            print(f"❌ Ошибка добавления лица {image_path}: {e}")
            return False

    # ---------- Управление камерой и потоками ----------

    def _open_camera(self):
        """Открытие камеры с автоматическим подбором транспорта."""
        is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'http://'))

        if is_rtsp:
            print(f"[CAMERA] Подключение к RTSP: {self.video_source}")
            transports = ['tcp', 'udp']

            for transport in transports:
                try:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
                    cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"[CAMERA] Успешно! Транспорт: {transport}")
                            return cap
                        cap.release()
                except Exception as e:
                    print(f"[CAMERA] Ошибка при проверке транспорта {transport}: {e}")
                    continue

            print("[CAMERA-ERROR] Не удалось подключиться к RTSP")
            return None
        else:
            # Для локальной камеры на macOS явно передаем cv2.CAP_AVFOUNDATION
            print(f"[CAMERA] Подключение к локальной веб-камере #{self.video_source}")
            import platform
            backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.video_source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap

    def start_threads(self):
        """Безопасный старт потоков с сохранением дескрипторов."""
        self.cap = self._open_camera()

        self.running = True
        self.processing = True

        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)

        self.capture_thread.start()
        self.process_thread.start()

    def stop_threads(self):
        """Корректная остановка: дожидаемся выхода потоков ДО освобождения self.cap."""
        self.running = False
        self.processing = False

        # 1. Ждем, пока поток захвата закончит cap.read() и остановится
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)

        # 2. Ждем, пока поток распознавания закончит работу
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.5)

        # 3. Безопасно освобождаем камеру только после полной остановки фоновых потоков
        with self.frame_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def change_source(self, new_source):
        """Переключение камеры без гонки потоков."""
        print(f"[CAMERA] Переключение источника на: {new_source}")
        self.stop_threads()

        # Очищаем очередь старых кадров
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass

        self.video_source = self._parse_source(new_source)
        self.start_threads()

    # ---------- Запись посещаемости ----------

    def save_attendance(self, name):
        from database import mark_attendance
        try:
            mark_attendance(student_name=name)
            print(f"✅ Отмечен в базе: {name}")
        except Exception as e:
            print(f"❌ Ошибка сохранения посещаемости: {e}")
