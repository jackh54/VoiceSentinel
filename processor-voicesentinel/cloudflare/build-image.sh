#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${WHISPER_MODEL:-Systran/faster-whisper-base}"

docker build \
  --platform linux/amd64 \
  --build-arg PREBAKE_WHISPER_MODEL=true \
  --build-arg WHISPER_MODEL="${MODEL}" \
  --build-arg WHISPER_DEVICE=cpu \
  --build-arg WHISPER_COMPUTE_TYPE=int8 \
  -f "${ROOT}/Dockerfile" \
  -t voicesentinel-processor:cf \
  "${ROOT}"
