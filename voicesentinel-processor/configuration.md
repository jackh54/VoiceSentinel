---
description: Processor config.json – pool flags, server, transcription, audio, processing, recordings, response, CORS, LLM profanity, report_buffer, console.
---

# Configuration

Config file: **`config.json`** in the processor directory. Override path with **`CONFIG_PATH`**. Start from **`config.example.json`**.

## Pool / multi-tenant (optional)

Top-level keys for **hosted pool** deployments (most self-hosted servers leave **`pool_server`** `false`):

| Key | Typical | Description |
|-----|---------|-------------|
| **pool_server** | `false` | When `true`, processor runs in pool mode (see internal deployment docs). |
| **pool_server_audit_log** | `logs/pooled_server_audit.jsonl` | Audit log path when pooling. |
| **pool_server_transcripts_dir** | `logs/pool_transcripts_by_license` | Transcript storage root when pooling. |

## Server (always review)

| Key | Example | Description |
|-----|---------|--------------|
| **host** | `"0.0.0.0"` | Bind address. |
| **port** | `28472` | HTTP + WebSocket port. Plugin **`processor_websocket_url`** must match (`ws://` / `wss://`). |
| **server_key** | `"YourSecret16CharsMin"` | Shared secret with the plugin in **CUSTOM** mode. 16–256 alphanumeric. Empty = accept any key (**not recommended**). PUBLIC pool processors use a fixed pool key on their side. |

## Transcription

| Key | Typical | Description |
|-----|---------|--------------|
| **model** | `"Systran/faster-whisper-base"` | Hugging Face model ID (see [Hugging Face models](huggingface-models.md)). |
| **language** | `"auto"` | `"auto"` or a fixed code (e.g. `"en"`). |
| **device** | `"cpu"` | `"cpu"` or `"cuda"`. |
| **compute_type** | `"int8"` | e.g. `int8` on CPU. |
| **timeout_seconds** | `30` | Max time per job before drop. |
| **cpu_threads** | `2` | Worker threads. |
| **huggingface_token** | `""` | Optional **`hf_…`** token for faster / higher-limit model downloads. |

## Audio

| Key | Default | Description |
|-----|---------|-------------|
| **min_audio_length_ms** | `50` | Shorter chunks skipped. |
| **max_audio_length_ms** | `30000` | Max clip length. |
| **sample_rate** | `16000` | Expected sample rate. |
| **channels** | `1` | Mono. |

## Processing queue

| Key | Default | Description |
|-----|---------|-------------|
| **queue_max_size** | `500` | Waiting jobs; when full, new work is dropped. Monitor **`processing_queue_size`** via **GET /stats**. |

## Recordings

| Key | Example | Description |
|-----|---------|--------------|
| **save_mode** | `"none"` | **`none`**, **`all`**, or **`flagged`**. |
| **save_path** | `"recordings/"` | WAV + metadata directory. |
| **retention_days** | `7` | File retention. |

## Response to plugin

**response** – What goes back on flagged paths:

| Key | Default | Description |
|-----|---------|-------------|
| **include_audio** | `false` | If `true`, flagged responses can include **base64 WAV** for Discord **`include_audio`** on the plugin. |

## CORS

**cors** – Relevant for **browser** calls to HTTP endpoints (**/health**, **/stats**). The Minecraft plugin does not need CORS. Tighten **`allow_origins`** if you expose APIs to a known web origin.

## LLM profanity (optional)

**llm_profanity** – Second-pass detection when word lists do not match.

| Key | Example | Description |
|-----|---------|--------------|
| **enabled** | `false` | |
| **provider** | `"ollama"` | **`openai`**, **`anthropic`**, **`ollama`**. |
| **api_key** | varies | Provider key or Ollama base URL. |
| **model** | `"llama2"` | Model id. |
| **timeout_seconds** | `15` | |
| **confidence_threshold** | `0.7` | |
| **strictness** | `"medium"` | **`strict`**, **`medium`**, **`lenient`**. |
| **max_concurrent_requests** | `3` | |
| **fallback_on_error** | `true` | If LLM fails, fall back without blocking the pipeline. |

## Report buffer (optional)

**report_buffer** – On-disk evidence for the plugin’s **GET /report/evidence** HTTP API.

| Key | Typical | Description |
|-----|---------|--------------|
| **enabled** | `false` | `true` to persist under **`path`**. |
| **path** | `"report_buffer/"` | Root directory. |
| **retention_seconds** | `604800` | Time to keep entries (e.g. 7 days). |
| **save_audio** | `false` | Also store audio when applicable. |

If **`enabled`** is **`false`**, the evidence endpoint returns **404**; the plugin still uses its **local transcript buffer** for books and the dashboard.

Plugin-side: [Voice reports & transcript buffer](../voicesentinel-plugin/voice-reports.md).

## Console

| Key | Default | Description |
|-----|---------|-------------|
| **log_transcripts** | `false` | Verbose transcript logging. |
| **live_display** | `true` | Periodic status line updates. |

<details>
<summary>Minimal self-hosted example</summary>

```json
{
  "pool_server": false,
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
    "timeout_seconds": 30,
    "cpu_threads": 2
  },
  "response": {
    "include_audio": false
  },
  "report_buffer": {
    "enabled": false,
    "path": "report_buffer/",
    "retention_seconds": 604800,
    "save_audio": false
  }
}
```

Match **`server_key`** to the plugin’s **`server_key`** in **CUSTOM** mode.
</details>
