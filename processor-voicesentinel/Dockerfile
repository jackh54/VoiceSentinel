# VoiceSentinel Processor - CPU Optimized Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including CPU optimization tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    opus-tools \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Whisper for faster container startup
RUN pip install --no-cache-dir openai-whisper

# Create required directories
RUN mkdir -p models

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.json ./

# CPU optimization environment variables
ENV OMP_NUM_THREADS=0
ENV MKL_NUM_THREADS=0
ENV OPENBLAS_NUM_THREADS=0
ENV NUMEXPR_NUM_THREADS=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Whisper cache optimization
ENV WHISPER_CACHE_DIR=/tmp/whisper_cache
RUN mkdir -p /tmp/whisper_cache

# Create entrypoint script for CPU optimization
RUN echo '#!/bin/bash\n\
# Auto-detect CPU cores in container\n\
CPU_CORES=$(nproc)\n\
echo "Container CPU cores: $CPU_CORES"\n\
\n\
# Set CPU optimization based on available cores\n\
export OMP_NUM_THREADS=$CPU_CORES\n\
export MKL_NUM_THREADS=$CPU_CORES\n\
export OPENBLAS_NUM_THREADS=$CPU_CORES\n\
export NUMEXPR_NUM_THREADS=$CPU_CORES\n\
\n\
echo "CPU optimization: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"\n\
\n\
# Start the application\n\
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use the CPU-optimized entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"] 