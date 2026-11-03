# linux/amd64 benchmark image. Run with --cpus 2 to mimic the shared 2-core production host.
FROM --platform=linux/amd64 python:3.12-slim
RUN pip install --no-cache-dir sherpa-onnx==2.0.1 numpy scipy soundfile jiwer psutil sentencepiece pypinyin
WORKDIR /spike
COPY *.py ./
ENV ST_MODELS=/models ST_DATA=/data ST_OUT=/out PYTHONUNBUFFERED=1
