# Running the processor

## With Docker

From `VoiceSentinel/processor-voicesentinel` (with **config.json** in place):

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

The compose file exposes port **28472** by default and mounts **config.json** and optionally a recordings directory. Healthcheck pings **/health** so orchestrators can see if it’s up.

## Without Docker (manual)

Activate your venv, then:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 28472
```

Use the same port as in **config.json** (**server.port**). To use a different port, pass e.g. `--port 9000` and set **server.port** in config to match so the plugin knows where to connect.

Run it in the background with **screen**, **tmux**, or a systemd service so it keeps running after you log out.

## Endpoints

- **GET /health** – Returns `{"status":"healthy","version":"3.0.0"}`. Use this for load balancers and health checks.
- **GET /stats** – Returns connection and queue info: **active_connections**, **transcriber_ready**, **processing_queue_size**.

The plugin talks over **WebSocket** at **/ws/&lt;client_id&gt;**. It sends **auth** first (with **server_key**), then **audio_chunk** messages. You don’t call the WebSocket URL from a browser for normal use; the plugin does it.

## First run

On first run the app downloads the Whisper model (from Hugging Face). That can take a few minutes and needs internet. After that it’s cached. If startup seems to hang, check the logs – it’s often still downloading.
