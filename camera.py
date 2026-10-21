import os
import time
import warnings
import cv2
import numpy as np
import threading
import pickle
from queue import Queue
from collections import deque, Counter

# Подавление предупреждений устаревания от scikit-image внутри insightface (SimilarityTransform.estimate)
warnings.filterwarnings("ignore", category=FutureWarning)


class DirectInsightFaceWrapper:
    """Прямой инференс через model_zoo.get_model в обход FaceAnalysis (Fallback)."""
    def __init__(self, det_model_path: str, rec_model_path: str, det_size=(640, 640), det_thresh=0.45):
        from insightface import model_zoo
        try:
            from insightface.app.common import Face
            self.Face = Face
        except ImportError:
            class Face(dict):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.__dict__ = self
            self.Face = Face

        print(f"🔧 [InsightFace Fallback] Загрузка детектора: {os.path.basename(det_model_path)}")
        self.detector = model_zoo.get_model(det_model_path, providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=-1, input_size=det_size, det_thresh=det_thresh)

        print(f"🔧 [InsightFace Fallback] Загрузка распознавателя: {os.path.basename(rec_model_path)}")
        self.recognizer = model_zoo.get_model(rec_model_path, providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=-1)

    def get(self, img):
        if img is None:
            return []
        bboxes, kpss = self.detector.detect(img, max_num=0, metric='default')
        if bboxes is None or len(bboxes) == 0:
            return []

        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = self.Face(bbox=bbox, kps=kps, det_score=det_score)
            self.recognizer.get(img, face)
            faces.append(face)
        return faces


class FaceRecognitionCamera:
    def __init__(self, path_images: str, video_source=0, delay=15.0):
        self.path_images = path_images
        self.video_source = self._parse_source(video_source)
        self.delay = delay

        # База известных лиц (512-D нормализованные векторы InsightFace)
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_detected_name = ""
        self.last_detection_time = 0
        self.cache_file = os.path.join(path_images, "face_cache_insightface.pkl")

        # Порог косинусного сходства (оптимизирован для камеры под потолком с углом обзора сверху)
        self.similarity_threshold = 0.48

        # Синхронизация потоков
        self.inference_lock = threading.Lock()
        self.frame_lock = threading.Lock()

        # Очередь для инференса (размер 1, drop-on-busy)
        self.frame_queue = Queue(maxsize=1)

        # Флаги управления потоками
        self.running = False
        self.processing = False

        # Сглаживание предсказаний (Temporal Smoothing)
        self.name_history = deque(maxlen=5)
        self.display_name = "Unknown"

        # Метрики производительности
        self.capture_fps = 0.0
        self.process_fps = 0.0

        # Видеодескрипторы и буферы кадров
        self.cap = None
        self.output_frame = None   # Кадр с нарисованными рамками для веб-стрима (MJPEG)
        self.raw_frame = None      # 100% чистый оригинальный кадр для эталонных снимков студентов

        # Кэш рамок для плавного отображения в потоке захвата
        self.cached_boxes = []
        self.cached_boxes_time = 0.0

        # Потоки
        self.capture_thread = None
        self.process_thread = None

        # Инициализация модели InsightFace
        self.app = self._init_insightface()

        # Загрузка базы студентов
        self.load_known_faces()

    @staticmethod
    def _parse_source(source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    # ---------- Инициализация InsightFace (Офлайн-режим) ----------

    def _init_insightface(self):
        """
        Инициализирует InsightFace (SCRFD детекция + ArcFace распознавание).
        Загрузка моделей строго из локальной папки в проекте без обращения к интернету.
        """
        from insightface.app import FaceAnalysis

        # 1. Потенциальные директории размещения моделей .onnx
        search_dirs = [
            os.path.abspath("./models/insightface/models/buffalo_l"),
            os.path.abspath("./models/models/buffalo_l"),
            os.path.abspath("./models/buffalo_l"),
            os.path.abspath("./models"),
            os.path.expanduser("~/.insightface/models/buffalo_l"),
        ]

        found_dir = None
        det_path = None
        rec_path = None

        for d in search_dirs:
            if not os.path.exists(d):
                continue

            onnx_files = [f for f in os.listdir(d) if f.lower().endswith('.onnx')]
            if not onnx_files:
                continue

            # Поиск модели детекции (начинается на det_)
            det_candidates = [os.path.join(d, f) for f in onnx_files if f.lower().startswith('det_')]
            # Поиск модели распознавания (w600k, r50, arcface, либо файл без det_/2d106/1k3d/genderage)
            rec_candidates = [
                os.path.join(d, f) for f in onnx_files
                if any(k in f.lower() for k in ('w600k', 'r50', 'arcface', 'glint'))
            ]
            if not rec_candidates:
                rec_candidates = [
                    os.path.join(d, f) for f in onnx_files
                    if not any(f.lower().startswith(p) for p in ('det_', '2d106', '1k3d', 'genderage'))
                ]

            if det_candidates:
                found_dir = d
                det_path = det_candidates[0]
                rec_path = rec_candidates[0] if rec_candidates else None
                break

        # 2. Валидация наличия файлов перед вызовом FaceAnalysis
        target_dir = os.path.abspath("./models/insightface/models/buffalo_l")
        if not found_dir or not det_path:
            current_files = os.listdir(target_dir) if os.path.exists(target_dir) else []
            error_msg = (
                f"\n❌ [InsightFace Error] Не найдены локальные веса моделей (.onnx)!\n"
                f"   Проверенная целевая папка: {target_dir}\n"
                f"   Файлы в папке сейчас: {current_files}\n"
                f"   Для офлайн-работы требуются файлы:\n"
                f"     1) Детектор: det_10g.onnx (или det_500m.onnx)\n"
                f"     2) Распознаватель: w600k_r50.onnx\n\n"
                f"   ИНСТРУКЦИЯ ДЛЯ ОФЛАЙН-РЕЖИМА:\n"
                f"   1. Скачайте архив buffalo_l.zip: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip\n"
                f"   2. Распакуйте его в папку: {target_dir}/\n"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        print(f"✅ [InsightFace] Найдены файлы моделей в: {found_dir}")
        print(f"   - Детектор: {os.path.basename(det_path)}")
        if rec_path:
            print(f"   - Распознаватель: {os.path.basename(rec_path)}")

        # 3. Вариант А: Инициализация через FaceAnalysis с корректным расчётом root и name
        try:
            # FaceAnalysis собирает путь: os.path.join(root, 'models', name, '*.onnx')
            parts = os.path.normpath(found_dir).split(os.sep)
            if len(parts) >= 2 and parts[-2] == "models":
                model_name = parts[-1]
                base_root = os.sep.join(parts[:-2])
            else:
                model_name = "buffalo_l"
                base_root = os.path.abspath("./models/insightface")

            print(f"🔄 [InsightFace] Инициализация FaceAnalysis(name='{model_name}', root='{base_root}')...")
            app = FaceAnalysis(
                name=model_name,
                root=base_root,
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider']
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print("🚀 [InsightFace] Модули успешно инициализированы через FaceAnalysis")
            return app

        except Exception as e:
            # 4. Вариант Б (Fallback): Прямой инференс через model_zoo.get_model
            print(f"⚠️ [InsightFace] Ошибка запуска через FaceAnalysis: {e}")
            print("🔄 [InsightFace] Запуск прямого Fallback через model_zoo.get_model...")
            if not rec_path:
                raise RuntimeError("Не найден файл модели распознавания (*.onnx) для fallback")

            fallback_app = DirectInsightFaceWrapper(
                det_model_path=det_path,
                rec_model_path=rec_path,
                det_size=(640, 640),
                det_thresh=0.45
            )
            print("🚀 [InsightFace] Модели успешно инициализированы через прямой Fallback (model_zoo)")
            return fallback_app

    # ---------- Загрузка и извлечение эмбеддингов лиц ----------

    def _extract_face_embedding(self, image_path: str):
        """Извлекает 512-мерный нормализованный вектор лица из изображения."""
        try:
            img = self._imread_unicode(image_path)
            if img is None:
                print(f"❌ Не удалось открыть файл: {image_path}")
                return None

            with self.inference_lock:
                faces = self.app.get(img)

            # Если лицо не найдено при стандартном освещении, применяем CLAHE
            if not faces:
                enhanced = self._enhance_contrast(img)
                with self.inference_lock:
                    faces = self.app.get(enhanced)

            if faces:
                # Если на снимке несколько лиц, берем самое крупное по площади bbox
                faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                target_face = faces[0]
                norm_emb = getattr(target_face, 'normed_embedding', None)
                if norm_emb is None or len(norm_emb) == 0:
                    norm_emb = target_face.embedding / np.linalg.norm(target_face.embedding)
                return norm_emb
            else:
                print(f"⚠️ Лицо не найдено на снимке: {image_path}")
                return None
        except Exception as e:
            print(f"❌ Ошибка извлечения эмбеддинга из {image_path}: {e}")
            return None

    def load_known_faces(self):
        """
        Загрузка векторов лиц: поддерживает как папки студентов (uploads/ФИО/*.jpg),
        так и одиночные файлы (uploads/ФИО.jpg).
        """
        if not os.path.exists(self.path_images):
            return

        new_encodings = []
        new_names = []

        print("🔄 [InsightFace] Загрузка базы студентов...")
        for item in sorted(os.listdir(self.path_images)):
            item_path = os.path.join(self.path_images, item)
            if item.startswith('.'):
                continue

            # 1. Если это папка студента с несколькими ракурсами (uploads/Иванов И.И./)
            if os.path.isdir(item_path):
                student_name = item
                for filename in sorted(os.listdir(item_path)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(item_path, filename)
                        emb = self._extract_face_embedding(img_path)
                        if emb is not None:
                            new_encodings.append(emb)
                            new_names.append(student_name)

            # 2. Одиночные файлы (uploads/Иванов И.И.jpg)
            elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                student_name = os.path.splitext(item)[0]
                emb = self._extract_face_embedding(item_path)
                if emb is not None:
                    new_encodings.append(emb)
                    new_names.append(student_name)

        with self.inference_lock:
            self.known_face_encodings = new_encodings
            self.known_face_names = new_names

        print(f"✅ [InsightFace] Загружено векторов: {len(self.known_face_encodings)} для студентов: {set(self.known_face_names)}")

    def load_fresh_faces(self):
        """Псевдоним для совместимости с маршрутами main.py."""
        self.load_known_faces()

    def add_single_face(self, image_path: str, name: str) -> bool:
        """
        Быстрое добавление одного нового ракурса студента без перезагрузки всей базы с диска.
        """
        try:
            emb = self._extract_face_embedding(image_path)
            if emb is not None:
                with self.inference_lock:
                    self.known_face_encodings.append(emb)
                    self.known_face_names.append(name)
                print(f"✅ [InsightFace] Добавлен ракурс для: {name}")
                return True
            else:
                print(f"⚠️ [InsightFace] Лицо не обнаружено на снимке: {image_path}")
                return False
        except Exception as e:
            print(f"❌ Ошибка добавления лица {image_path}: {e}")
            return False

    # ---------- Поток захвата видеокадров (Capture Thread) ----------

    def capture_frames(self):
        """
        Высокоскоростной поток захвата кадров с камеры (25-30 FPS).
        Развязан с нейросетью: захват не ждёт инференса, видеопоток никогда не лагает.
        """
        print("📹 Поток захвата запущен")
        frame_count = 0
        fps_start = time.time()

        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            # 1. Сохраняем нетронутый оригинальный кадр для снимков
            with self.frame_lock:
                self.raw_frame = frame.copy()

            # 2. Отправляем копию кадра на инференс, ТОЛЬКО если нейросеть свободна (drop-on-busy)
            if self.frame_queue.empty():
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            # 3. Формируем кадр для браузера (MJPEG-стрим) с наложением закэшированных рамок
            display = frame.copy()
            now = time.time()

            with self.frame_lock:
                # Отображаем рамки, если они получены не более 0.8 сек назад
                if self.cached_boxes and (now - self.cached_boxes_time < 0.8):
                    for box in self.cached_boxes:
                        left, top, right, bottom = box['bbox']
                        label = box['label']
                        color = box['color']

                        # Рамка вокруг лица
                        cv2.rectangle(display, (left, top), (right, bottom), color, 2)

                        # Подложка под подпись
                        cv2.rectangle(display, (left, bottom - 26), (right, bottom), color, cv2.FILLED)
                        cv2.putText(display, label, (left + 6, bottom - 7),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                # Метрики FPS в левом верхнем углу
                cv2.putText(display, f"Cap: {self.capture_fps:.1f} FPS | Proc: {self.process_fps:.1f} FPS",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                self.output_frame = display

            # Расчёт FPS захвата
            frame_count += 1
            if frame_count >= 30:
                elapsed = time.time() - fps_start
                if elapsed > 0:
                    self.capture_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

    # ---------- Поток инференса нейросети (Recognition Thread) ----------

    def process_frames(self):
        """
        Асинхронный рабочий поток распознавания: SCRFD детекция + ArcFace эмбеддинги.
        """
        print("⚙️ Поток обработки запущен (InsightFace SCRFD + ArcFace)")
        frame_count = 0
        fps_start = time.time()

        while self.processing:
            try:
                try:
                    frame = self.frame_queue.get(timeout=0.05)
                except Exception:
                    continue

                # Инференс нейросети на чистом кадре
                with self.inference_lock:
                    faces = self.app.get(frame)

                detected_boxes = []
                recognized_student = None

                if faces:
                    # Подготовка матрицы известных векторов для векторного перемножения
                    known_matrix = None
                    known_names = []
                    with self.inference_lock:
                        if self.known_face_encodings:
                            known_matrix = np.array(self.known_face_encodings, dtype=np.float32)
                            known_names = list(self.known_face_names)

                    h, w = frame.shape[:2]

                    for face in faces:
                        x1, y1, x2, y2 = [int(v) for v in face.bbox]
                        left = max(0, x1)
                        top = max(0, y1)
                        right = min(w, x2)
                        bottom = min(h, y2)

                        face_name = "Unknown"
                        is_recognized = False

                        if known_matrix is not None and len(known_matrix) > 0:
                            norm_emb = getattr(face, 'normed_embedding', None)
                            if norm_emb is None or len(norm_emb) == 0:
                                norm_emb = face.embedding / np.linalg.norm(face.embedding)

                            # Косинусное сходство через скалярное произведение (векторы уже L2-нормализованы)
                            sims = np.dot(known_matrix, norm_emb)
                            best_idx = int(np.argmax(sims))
                            best_score = float(sims[best_idx])

                            # Порог сходства с учётом ракурса сверху камеры под потолком
                            if best_score >= self.similarity_threshold:
                                face_name = known_names[best_idx]
                                is_recognized = True
                                if recognized_student is None:
                                    recognized_student = face_name

                        color = (0, 255, 0) if is_recognized else (0, 165, 255)
                        detected_boxes.append({
                            'bbox': (left, top, right, bottom),
                            'label': face_name,
                            'color': color,
                            'recognized': is_recognized
                        })

                # Сглаживание предсказаний имени
                current_detected = recognized_student if recognized_student else "Unknown"
                self.name_history.append(current_detected)
                if self.name_history:
                    self.display_name = Counter(self.name_history).most_common(1)[0][0]

                # Фиксация посещаемости в БД
                if self.display_name != "Unknown":
                    now = time.time()
                    if (self.display_name != self.last_detected_name or
                            now - self.last_detection_time > self.delay):
                        self.save_attendance(self.display_name)
                        self.last_detected_name = self.display_name
                        self.last_detection_time = now

                # Передаем координаты рамок в поток захвата под lock
                with self.frame_lock:
                    self.cached_boxes = detected_boxes
                    self.cached_boxes_time = time.time()

                # Расчёт FPS инференса
                frame_count += 1
                if frame_count >= 15:
                    elapsed = time.time() - fps_start
                    if elapsed > 0:
                        self.process_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                print(f"❌ Ошибка в потоке обработки: {e}")
                time.sleep(0.05)

    # ---------- Отдача кадра и снимков ----------

    def get_jpeg_frame(self) -> bytes:
        """
        Возвращает сжатый кадр с рамками для HTTP MJPEG стрима.
        Используется оптимизированное сжатие качества 75 для минимальной нагрузки на сеть и CPU.
        """
        with self.frame_lock:
            if self.output_frame is None:
                return None
            ok, jpeg = cv2.imencode('.jpg', self.output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return None
            return jpeg.tobytes()

    def save_snapshot(self, file_path: str) -> bool:
        """
        Сохранение ЧИСТОГО кадра из оперативной памяти без наложенных рамок и текста.
        Гарантирует чистоту эталонного снимка студента.
        """
        try:
            with self.frame_lock:
                if self.raw_frame is not None:
                    frame_to_save = self.raw_frame.copy()
                elif self.output_frame is not None:
                    frame_to_save = self.output_frame.copy()
                else:
                    return False

            return self._imwrite_unicode(file_path, frame_to_save)
        except Exception as e:
            print(f"❌ Ошибка сохранения снимка: {e}")
            return False

    # ---------- Управление камерой и потоками ----------

    def _open_camera(self):
        """Открытие камеры с минимальной задержкой буферизации RTSP."""
        is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'http://'))

        if is_rtsp:
            print(f"[CAMERA] Подключение к RTSP: {self.video_source}")
            # TCP транспорт предпочтителен для Full HD во избежание артефактов и выпадения пакетов
            transports = ['tcp', 'udp']

            for transport in transports:
                try:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        f"rtsp_transport;{transport}|fflags;nobuffer|max_delay;500000|flags;low_delay"
                    )
                    cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            print(f"[CAMERA] Успешно! Транспорт: {transport}")
                            return cap
                        cap.release()
                except Exception as e:
                    print(f"[CAMERA] Ошибка проверки транспорта {transport}: {e}")
                    continue

            print("[CAMERA-ERROR] Не удалось подключиться к RTSP")
            return None
        else:
            print(f"[CAMERA] Подключение к локальной веб-камере #{self.video_source}")
            import platform
            backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.video_source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap

    def start_threads(self):
        """Безопасный запуск фоновых потоков."""
        self.cap = self._open_camera()

        self.running = True
        self.processing = True

        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)

        self.capture_thread.start()
        self.process_thread.start()

    def stop_threads(self):
        """
        Корректная остановка: дожидаемся завершения потоков через join(timeout) ДО release().
        Исключает падения SIGSEGV / аварийные завершения.
        """
        self.running = False
        self.processing = False

        if hasattr(self, 'capture_thread') and self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)

        if hasattr(self, 'process_thread') and self.process_thread is not None and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.5)

        with self.frame_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def change_source(self, new_source):
        """Переключение видеоисточника без гонки потоков."""
        print(f"[CAMERA] Переключение источника на: {new_source}")
        self.stop_threads()

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

    # ---------- Утилиты для работы с изображениями и кириллицей ----------

    @staticmethod
    def _enhance_contrast(image_bgr):
        """Контрастирование CLAHE по L-каналу для компенсации теней от потолочного освещения."""
        try:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            return image_bgr

    @staticmethod
    def _imread_unicode(path: str):
        """Надёжное чтение изображения по пути с кириллицей (Windows Server / Linux)."""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        return cv2.imread(path)

    @staticmethod
    def _imwrite_unicode(path: str, img) -> bool:
        """Надёжное сохранение изображения по пути с кириллицей (Windows Server / Linux)."""
        try:
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = ".jpg"
            ok, buf = cv2.imencode(ext, img)
            if ok:
                buf.tofile(path)
                return True
        except Exception:
            pass
        return bool(cv2.imwrite(path, img))
