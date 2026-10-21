# ================= ЭТАП 1: Сборка wheel-пакетов =================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Собираем готовые wheel-пакеты в отдельную папку
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt


# ================= ЭТАП 2: Чистый итоговый образ =================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ставим только необходимые runtime-библиотеки:
# - libglib2.0-0 для OpenCV
# - libgomp1 для многопоточного OpenMP инференса ONNX Runtime на CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем собранные wheels из builder и устанавливаем их
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Копируем код проекта (включая папку models/ с локальными весами InsightFace)
COPY . .

# Создаем директории для загрузки фото и оффлайн-моделей
RUN mkdir -p uploads models/insightface/models/buffalo_l

EXPOSE 8000

CMD ["python", "main.py"]
