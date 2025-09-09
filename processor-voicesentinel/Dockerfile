FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    opus-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai-whisper

RUN mkdir -p models

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config.json ./

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WHISPER_CACHE_DIR=/tmp/whisper_cache

RUN mkdir -p /tmp/whisper_cache

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:28472/health || exit 1

EXPOSE 28472

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "28472", "--workers", "1"] 