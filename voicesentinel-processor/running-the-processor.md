# Running the processor

## With Docker

From **`VoiceSentinel/processor-voicesentinel`** (with **`config.json`** in place):

```bash
docker compose up -d
```

Logs (follow):

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

The compose file exposes port **28472** by default and mounts **`config.json`** (and optionally recordings). Healthcheck hits **GET /health**.

## Without Docker (manual)

Activate your venv, then:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 28472
```

Use the same port as **`server.port`** in **`config.json`**. For another port, pass **`--port`** and update **`config.json`** so the plugin’s **`processor_websocket_url`** matches.

Run under **screen**, **tmux**, or **systemd** for production.

## HTTP endpoints

- **GET /health** — JSON with **`status`** and **`version`** (processor build string). Use for load balancers and monitoring.
- **GET /stats** — Connection and queue metrics (**active_connections**, **transcriber_ready**, **processing_queue_size**, etc.).

## WebSocket

The plugin connects over WebSocket (path includes a client id). It sends **auth** (with **server_key** in **CUSTOM** deployments), then **audio_chunk** messages. Normal gameplay does not use a browser against the WebSocket URL.

## First run

The first start downloads the Whisper model from Hugging Face (can take several minutes; requires internet). Later starts use the cache. Set **`transcription.huggingface_token`** for faster, more reliable downloads ([Hugging Face models](huggingface-models.md)).
