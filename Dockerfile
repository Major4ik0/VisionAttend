# ================= ЭТАП 1: Сборка wheel-пакетов =================
FROM python:3.11-slim AS builder

# Устанавливаем компиляторы только для сборки dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
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

# Ставим только runtime-библиотеки (без gcc, cmake и dev-заголовков)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем собранные wheels из builder и устанавливаем их
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Копируем код проекта
COPY . .

# Создаем директорию для загрузки фото
RUN mkdir -p uploads

EXPOSE 8000

CMD ["python", "main.py"]