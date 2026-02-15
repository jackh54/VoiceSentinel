---
description: Choose the right faster-whisper model and how much RAM to allocate by player count and concurrent speakers.
---

# Hugging Face models and sizing

The processor uses **faster-whisper** (Systran) models from Hugging Face for speech-to-text. You pick one model in **transcription.model** in `config.json`. Smaller models use less RAM and are faster but less accurate; larger models need more resources and handle accents/noise better.

## Model comparison

All of these are used with **compute_type** `int8` in the processor (default), which keeps memory lower than full precision.

| Model (config value) | Min RAM (CPU, int8) | Min VRAM (GPU, int8) | Relative speed | Accuracy |
|----------------------|---------------------|----------------------|----------------|----------|
| `Systran/faster-whisper-tiny` | ~0.5 GB | ~0.5 GB | Fastest | Lowest – OK for clear English only |
| `Systran/faster-whisper-base` | ~1 GB | ~1 GB | Very fast | Good for clear speech, single language |
| `Systran/faster-whisper-small` | ~2 GB | ~1 GB | Fast | Better accents and noise |
| `Systran/faster-whisper-medium` | ~5 GB | ~2 GB | Moderate | Strong accuracy, multiple languages |
| `Systran/faster-whisper-large-v2` | ~8 GB | ~3 GB | Slower | High accuracy |
| `Systran/faster-whisper-large-v3` | ~10 GB | ~4 GB | Slowest | Best accuracy, best for mixed/noisy audio |

RAM/VRAM are approximate; actual use depends on `cpu_threads`, queue depth, and OS. Add headroom for the rest of the process (Python, HTTP, buffers).

## Sizing: allocation and capacity

Use this as a starting point. “Concurrent speakers” = how many people can be talking at once and still get transcripts without the queue backing up. “Typical server size” = rough player count if only a few talk at once; bursty voice = plan for higher concurrency.

| Model | Allocate RAM (CPU) | Rough concurrent speakers | Typical server size (players) |
|-------|--------------------|---------------------------|-------------------------------|
| **tiny** | 2 GB | 4–6 | Small (10–20) |
| **base** | 4 GB | 3–5 | Small–medium (20–40) |
| **small** | 6 GB | 2–4 | Medium (40–80) |
| **medium** | 8 GB | 1–3 | Medium–large (60–100) |
| **large-v2** | 12 GB | 1–2 | Large (80–150) |
| **large-v3** | 16 GB | 1–2 | Large (80–150) |

- **Allocate RAM** = recommended minimum for the processor (including model + queue + overhead). On shared hosts, give the processor at least this much.
- **Concurrent speakers** = approximate. If many players talk at once, use a smaller model or more RAM/CPU, or increase **processing.queue_max_size** and accept some delay.
- **Typical server size** = ballpark. Assumes “a few people talking at a time.” If your server has 50 people all in voice at once, aim for the concurrent-speakers column and possibly a smaller/faster model.

{% hint style="info" %}
With a **GPU** (`device: "cuda"`), the same model uses less CPU and can handle more concurrent jobs. You can often use **large-v2** or **large-v3** at 4–6 GB VRAM and allocate less RAM to the server.
{% endhint %}

## Choosing a model

- **Small server, low traffic** – Start with **base** (4 GB RAM). Good balance of speed and accuracy for clear English.
- **Lots of accents or background noise** – Prefer **small** or **medium**; **large-v2** / **large-v3** if you have the RAM/GPU.
- **Multiple languages** – **medium** or **large-v3** with `language: "auto"`.
- **Max accuracy, powerful box** – **large-v3** with 16 GB RAM or a GPU with 4+ GB VRAM.
- **Limited RAM (e.g. 2 GB)** – **tiny** or **base**; expect lower accuracy on noisy or non-English audio.

Set the model in [Configuration](configuration.md) under **transcription.model** (e.g. `Systran/faster-whisper-base`). Restart the processor after changing it; the first run will download the model from Hugging Face if needed.
