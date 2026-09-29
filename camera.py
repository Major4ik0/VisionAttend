import os
import time
import cv2
import numpy as np
import face_recognition
import threading
from queue import Queue
from database import mark_attendance

class FaceRecognitionCamera:
    def __init__(self, path_images: str, video_source=0, delay=15.0):
        self.path_images = path_images
        self.video_source = self._parse_source(video_source)
        self.delay = delay
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_detected_name = ""
        self.last_detection_time = 0
        
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.processing = False
        self.output_frame = None
        self.lock = threading.Lock()
        self.cap = None
        
        self.load_known_faces()

    def _parse_source(self, source):
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def load_known_faces(self):
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images, exist_ok=True)
            return
        self.load_fresh_faces()

    def load_fresh_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []

        for filename in os.listdir(self.path_images):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(self.path_images, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.known_face_encodings.append(encoding[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"Ошибка при загрузке {filename}: {e}")

    def capture_frames(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            
            if isinstance(self.video_source, int):
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (640, 380))
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except:
                    pass

    def process_frames(self):
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    name = "Unknown"
                    color = (0, 165, 255)

                    if self.known_face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]
                            color = (0, 255, 0)

                            current_time = time.time()
                            if (name != self.last_detected_name or 
                                current_time - self.last_detection_time > self.delay):
                                mark_attendance(student_name=name)
                                self.last_detected_name = name
                                self.last_detection_time = current_time

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                with self.lock:
                    self.output_frame = frame.copy()

    def get_jpeg_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.output_frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def change_source(self, new_source):
        self.running = False
        self.processing = False
        time.sleep(0.3)

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

        t1 = threading.Thread(target=self.capture_frames, daemon=True)
        t2 = threading.Thread(target=self.process_frames, daemon=True)
        t1.start()
        t2.start()
        return True
