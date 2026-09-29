FROM python:3.11-slim

# Установка системных зависимостей для dlib, opencv и ffmpeg/rtsp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование и установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создаем директории для загрузок и конфигов
RUN mkdir -p uploads

EXPOSE 8000

CMD ["python", "main.py"]
