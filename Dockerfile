FROM python:3.11-slim

# Системные зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python зависимости (кэшируем отдельным слоем)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY api/ ./api/
COPY src/ ./src/
COPY weights/ ./weights/
COPY zones.json zones_02.json ./

# Render даёт порт через $PORT. Fallback — 8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
