# VoiceSentinel – Real-Time Voice Moderation Plugin

**Author:** pandadevv  
**Voice Platform:** [Simple Voice Chat](https://voicechat.modrepo.de/)  
**Server Type:** Minecraft (Folia – region-based async threading)  
**License:** MIT

---

## 📘 Overview

VoiceSentinel is a comprehensive voice moderation system built for Minecraft servers running Folia & Paper. It provides real-time detection and filtering of inappropriate voice chat content using the Simple Voice Chat mod. The system captures audio from players, processes it asynchronously, and sends it to a Dockerized backend for transcription and content filtering.

### Key Features

- **Real-time voice monitoring** with Simple Voice Chat integration
- **Folia-safe architecture** with proper async handling
- **Dockerized processing backend** for scalability
- **Pterodactyl Support for backend** for easy integration
- **Multiple transcription engines** (Whisper.cpp, Vosk)
- **Configurable profanity filtering** with customizable word lists
- **Staff alerting system** for flagged content
- **Low-latency processing** (< 300ms expected roundtrip)
- **Thread-safe session management** per player

---

## 🏗️ Architecture

### Plugin Component (Java - Folia Safe)
- Intercepts Opus voice packets via `MicrophonePacketEvent`
- Buffers per-player audio in ~2-second windows
- Encodes and dispatches audio + metadata to external processor
- Matches responses by `session_id`
- Uses `RegionScheduler` for safe UI feedback

### Processor Component (Python + Docker)
- Accepts JSON payload with metadata + Opus audio
- Decodes audio to PCM/WAV (via `ffmpeg`)
- Transcribes via `whisper.cpp` or `vosk`
- Runs profanity filter on transcript
- Returns JSON response to plugin

---

# to be written...


**Built with ❤️ by pandadevv**
