---
description: Choose the right faster-whisper model and how much RAM to allocate by player count and concurrent speakers.
---

# Hugging Face models and sizing

The processor uses **faster-whisper**-compatible models from Hugging Face for speech-to-text. You pick one model in **transcription.model** in `config.json`. The table below lists common Systran models as a starting point – you don’t have to use only those. You can browse [Hugging Face models tagged for faster-whisper](https://huggingface.co/models?other=faster-whisper) (or the [Systran organization](https://huggingface.co/Systran)) and set **transcription.model** to any model ID that works with faster-whisper (e.g. `Systran/faster-whisper-base` or another repo). Smaller models use less RAM and are faster but less accurate; larger models need more resources and handle accents/noise better.

## Hugging Face token (faster downloads)

The first time you run a model, the processor downloads it from Hugging Face. Unauthenticated downloads are rate-limited and can be slow. Adding a **Hugging Face access token** gives you higher limits and faster downloads.

**How to get a token:**

1. Sign up or log in at [huggingface.co](https://huggingface.co/join).
2. Open **Settings** (profile menu → Settings).
3. Go to **Access Tokens** ([direct link](https://huggingface.co/settings/tokens)).
4. Click **New token**, name it (e.g. `voicesentinel-processor`), choose **Read** (no need for write).
5. Copy the token (starts with `hf_`).

**Use it in the processor:**  
Put the token in `config.json` under **transcription** as **huggingface_token**:

```json
"transcription": {
  "model": "Systran/faster-whisper-base",
  "huggingface_token": "hf_YourTokenHere"
}
```

Leave **huggingface_token** out or set it to `""` if you don’t use a token; downloads will still work but may be slower. Don’t share or commit your token.

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

The chart below is for **specific specs**: the “Allocate RAM” value for that row **plus** a **moderate CPU** (e.g. 4 cores / 4–8 threads, as set in **transcription.cpu_threads**). Concurrent capacity is limited by CPU throughput, not just RAM – so with more cores/threads you can handle **more** people talking at once than the table shows. Use the table as a baseline for that spec.

| Model | Allocate RAM (CPU) | Concurrent speakers (4–8 threads) | Typical server size (players) |
|-------|--------------------|-----------------------------------|-------------------------------|
| **tiny** | 2 GB | 4–8 | Small (10–20) |
| **base** | 4 GB | 3–6 | Small–medium (20–50) |
| **small** | 6 GB | 2–5 | Medium (40–80) |
| **medium** | 8 GB | 2–4 | Medium–large (60–100) |
| **large-v2** | 12 GB | 1–3 | Large (80–150) |
| **large-v3** | 16 GB | 1–2 | Large (80–150) |

- **Allocate RAM** = recommended minimum for the processor (model + queue + overhead) for that model.
- **Concurrent speakers** = how many people talking at once the processor can keep up with **at that RAM and ~4–8 CPU threads**. More threads = higher concurrency; fewer threads = lower. Scale up **cpu_threads** (and RAM if needed) for more simultaneous voice.
- **Typical server size** = rough total players when only a subset are in voice at once. If everyone is in voice and talking a lot, size by the concurrent-speakers column instead (or use a smaller/faster model / more CPU).

{% hint style="info" %}
With a **GPU** (`device: "cuda"`), the same model uses less CPU and can handle more concurrent jobs. You can often use **large-v2** or **large-v3** at 4–6 GB VRAM and allocate less RAM to the server.
{% endhint %}

## Choosing a model

- **Small server, low traffic** – Start with **base** (4 GB RAM). Good balance of speed and accuracy for clear English.
- **Lots of accents or background noise** – Prefer **small** or **medium**; **large-v2** / **large-v3** if you have the RAM/GPU.
- **Multiple languages** – **medium** or **large-v3** with `language: "auto"`.
- **Max accuracy, powerful box** – **large-v3** with 16 GB RAM or a GPU with 4+ GB VRAM.
- **Limited RAM (e.g. 2 GB)** – **tiny** or **base**; expect lower accuracy on noisy or non-English audio.

Set the model in [Configuration](configuration.md) under **transcription.model** (e.g. `Systran/faster-whisper-base`). You can use any faster-whisper-compatible model from [Hugging Face](https://huggingface.co/models?other=faster-whisper) – just use its repo ID (e.g. `org/model-name`). Restart the processor after changing it; the first run will download the model from Hugging Face (faster if you set **huggingface_token** above).
