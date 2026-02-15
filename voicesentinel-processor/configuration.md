---
description: Processor config.json reference – server, transcription, audio, processing, recordings, CORS, and optional LLM profanity.
---

# Configuration

Config file: **config.json** in the processor folder. You can point the app at a different path with the **CONFIG_PATH** environment variable. The example file is **config.example.json** – copy it to **config.json** and edit.

## Must-set

**server** – How the processor listens and how the plugin authenticates.

| Key | Example | Description |
|-----|---------|--------------|
| **host** | `"0.0.0.0"` | Bind address. Use `0.0.0.0` to accept connections from other machines. |
| **port** | `28472` | Port for HTTP and WebSocket. Plugin uses this in **processor_websocket_url** (e.g. `ws://host:28472`). |
| **server_key** | `"YourSecretKey16CharsMin"` | Shared secret. Must match the plugin’s **server_key** in config.yml. 16–256 letters/numbers. Leave empty to accept any key (not recommended). |

## Transcription

**transcription** – Model and how long to wait.

| Key | Typical | Description |
|-----|---------|-------------|
| **model** | `"Systran/faster-whisper-base"` | Whisper model. `base` is faster/smaller, `small`/`medium`/`large-v2`/`large-v3` are heavier but more accurate. |
| **language** | `"auto"` | `"auto"` for auto-detect, or e.g. `"en"` to force English. |
| **device** | `"cpu"` | `"cpu"` or `"cuda"` if you have a GPU. |
| **compute_type** | `"int8"` | `int8` is common for CPU. |
| **timeout_seconds** | `30` | Max time per transcription; after that the job is dropped. |
| **cpu_threads** | `2` | Worker threads. Increase on beefy machines. |

## Audio limits

**audio** – What the processor accepts.

| Key | Default | Description |
|-----|---------|-------------|
| **min_audio_length_ms** | `50` | Chunks shorter than this are skipped. |
| **max_audio_length_ms** | `30000` | Chunks longer than 30 seconds are rejected. |
| **sample_rate** | `16000` | Expected sample rate. |
| **channels** | `1` | Mono. |

## Processing queue

**processing** – Backpressure.

| Key | Default | Description |
|-----|---------|-------------|
| **queue_max_size** | `500` | Max jobs waiting. When full, new recordings are dropped and the plugin gets an empty transcript. |

Increase this if you have bursts of traffic; you can also check **/stats** for **processing_queue_size**.

## Recordings (optional)

**recordings** – Saving WAVs (e.g. for moderation or compliance).

| Key | Example | Description |
|-----|---------|-------------|
| **save_mode** | `"none"` | `"none"` = don’t save. `"all"` = save everything. `"flagged"` = only when something is flagged. |
| **save_path** | `"recordings/"` | Directory for WAV + metadata files. |
| **retention_days** | `7` | Delete files older than this. |

Check your local laws and privacy policy before saving voice. Prefer `"none"` or `"flagged"` if you’re not sure.

## CORS

**cors** – Only matters if you call the HTTP endpoints (e.g. /health, /stats) from a browser. Typical:

```json
"cors": {
  "allow_origins": ["*"],
  "allow_credentials": true,
  "allow_methods": ["*"],
  "allow_headers": ["*"]
}
```

Tighten **allow_origins** if you have a known front-end URL.

## LLM profanity (optional & beta)

**llm_profanity** – Optional second pass: when no word-list match is found, an LLM can still flag the transcript. Disabled by default.

| Key | Example | Description |
|-----|---------|-------------|
| **enabled** | `false` | Turn on to use the LLM. |
| **provider** | `"ollama"` | `"openai"`, `"anthropic"`, or `"ollama"`. |
| **api_key** | `"http://localhost:11434"` | API key for OpenAI/Anthropic, or Ollama base URL. |
| **model** | `"llama2"` | Model name. For Ollama, use a model you’ve pulled. |
| **timeout_seconds** | `15` | Request timeout. |
| **confidence_threshold** | `0.7` | How confident the LLM must be to flag. |
| **strictness** | `"medium"` | `"strict"`, `"medium"`, or `"lenient"`. |

## Console

**console** – Logging and live stats.

| Key | Default | Description |
|-----|---------|-------------|
| **log_transcripts** | `false` | Log every transcript to the console. |
| **live_display** | `true` | Periodic live-updating status in the console. |

<details>
<summary>Minimal config example</summary>

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 28472,
    "server_key": "YourSharedSecretKey16CharsMin"
  },
  "transcription": {
    "model": "Systran/faster-whisper-base",
    "language": "auto",
    "device": "cpu",
    "compute_type": "int8",
    "timeout_seconds": 30
  },
  "audio": {
    "min_audio_length_ms": 50,
    "max_audio_length_ms": 30000
  },
  "processing": {
    "queue_max_size": 500
  },
  "recordings": {
    "save_mode": "none",
    "save_path": "recordings/",
    "retention_days": 7
  }
}
```

Set **server_key** to the same value you put in the plugin’s config.yml. Everything else has defaults in code if you omit it.
</details>
