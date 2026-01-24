FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p models recordings /tmp/whisper_cache

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app/ ./app/
RUN touch app/__init__.py 2>/dev/null || true

COPY config.example.json ./config.json

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TMPDIR=/tmp
ENV PIP_NO_CACHE_DIR=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:28472/health || exit 1

EXPOSE 28472

VOLUME ["/app/recordings", "/tmp/whisper_cache"]

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "28472", "--workers", "1"]
