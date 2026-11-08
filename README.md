# VoiceSentinel

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Minecraft](https://img.shields.io/badge/Minecraft-1.21+-green.svg)](https://www.minecraft.net/)

VoiceSentinel is a real-time voice moderation system for Minecraft servers using **Simple Voice Chat** or **PlasmoVoice**.

This public repo holds the self-hostable speech processor. The Minecraft plugin and its On-Server speech engine are closed source and [sold on BuiltByBit](https://builtbybit.com/voicesentinel).

| Folder | What it is |
|---|---|
| `processor-voicesentinel/` | Python processor for the public pool and self-hosting. |

The On-Server engine (`LOCAL` mode, no Docker or Python needed) is downloaded by the plugin automatically.

Three ways to run it: **On-Server** (`LOCAL`), **Public Pool** (`PUBLIC`) or **Self-Hosted** (`CUSTOM`).

## Features

### Core Functionality
- Real-time voice transcription using Faster Whisper (Systran/faster-whisper-large-v3)
- Multi-language support with automatic detection
- Advanced profanity detection with customizable word lists
- Configurable audio processing
- Rate limiting and authentication
- Queue-based processing for high-volume voice chat
- Recording save functionality (all, flagged, or none)

### Technical Features
- Asynchronous processing
- Docker support
- CPU-optimized for cost-effective hosting
- WebSocket-based communication
- Automatic cleanup of temporary files
- Health monitoring

### Minecraft Integration
- Folia & Paper compatibility
- Simple Voice Chat integration (default)
- PlasmoVoice integration (`voice_backend: PLASMOVOICE` or `AUTO`)

### Live monitoring dashboard (plugin v3.3.1+)
- "Live monitoring" tab in the web dashboard: live transcript stream, flagged-event feed, and per-player voice history.
- A standard part of the plugin — the whole plugin requires a valid license, so these panels work on any licensed install. Individual panels can be toggled in `config.yml`.

### Voice reports (plugin v2.2.1+)
- In-memory rolling transcript buffer per player (all final transcripts, not only flagged lines).
- `/reportvoice <player> [minutes]` (or a chest GUI when `voice_report.interface` is `gui`) notifies staff with `voicesentinel.report.review`. Staff see a compact chat line with a clickable **Open book** action (and the same command printed: `/viewreport <player> <minutes>` matching the report window).
- `/viewreport <player> [minutes]` opens a **written book** with the local buffer plus optional processor evidence—the same payload as the report, formatted for reading. Optional `voice_report.staff_chat_full_transcript: true` restores dumping the full plain text in chat for staff who want it.
- Optional processor retention: enable `report_buffer.enabled` on the processor (if it is off, `GET /report/evidence` returns **404** with a hint in the JSON body). Call `GET /report/evidence?player=<name>&seconds=<n>` with `X-License-Key` (same as the plugin) and **`X-Server-Key`** matching WebSocket auth (the plugin sends the **public pool key** in PUBLIC mode and `server_key` from `config.yml` in CUSTOM mode). Stored transcripts live under `report_buffer/<license_sha256>/<server_key_sha256>/` so **different licenses never share data**. **Different Minecraft servers with the same license** are isolated only if they use **different** `server_key` values (typical for self-hosted); on a **public pool** everyone shares the same pool `server_key`, so the second path segment is identical for all pool users under that license (still isolated from other licenses). Use HTTPS in front of public or shared processors.

## Requirements

### Server Requirements
- Java 21+ for the Minecraft plugin
- Python 3.8+ for the processor
- Minecraft 1.21+ Folia or Paper
- Simple Voice Chat **or** PlasmoVoice (not both)
- Self-hosted processor: 4GB RAM minimum (8GB+ recommended)
- On-Server Mode: about 1GB free RAM for English, about 2.5GB for multi-language
- Multi-core CPU

### Dependencies
- Backend: FastAPI, Faster Whisper, NumPy, WebSockets

## Quick Start

### 1. Backend Setup

```bash
git clone https://github.com/jackh54/VoiceSentinel.git
cd VoiceSentinel/processor-voicesentinel

cp config.example.json config.json
# Edit config.json with your settings

docker compose up -d

# View logs
docker compose logs -f
```

### 2. Plugin Installation

Install and configure the plugin. Set the same server-key in both processor & plugin configuration.

### 3. Configuration

#### Backend Configuration (config.json)

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 28472,
    "server_key": "your-secure-key-here"
  },
  "transcription": {
    "model": "Systran/faster-whisper-large-v3",
    "language": "auto",
    "device": "cpu",
    "compute_type": "int8"
  },
  "recordings": {
    "save_mode": "none",
    "save_path": "recordings/",
    "retention_days": 7
  }
}
```

Recording save modes:
- `none` - Do not save recordings
- `all` - Save all recordings
- `flagged` - Save only flagged recordings

## Cloudflare Containers (managed alternative)

Cloudflare Containers is an **optional** managed deployment path for the processor. Self-hosting via Docker Compose (above) remains the default and fully supported workflow — no config or behavior changes are required for existing installs.

For prerequisites, secrets, R2 bindings, sizing, and deploy steps, see [processor-voicesentinel/cloudflare/README.md](processor-voicesentinel/cloudflare/README.md).

### Privacy & Legal Notice

**Important:** When using the `recordings` configuration (specifically `save_mode`, `save_path`, and `retention_days`), server operators must comply with all applicable local, state, and federal laws regarding voice recording and data privacy.

**Legal Requirements:**
- **Consent Laws:** You must comply with all-party consent laws where applicable. In jurisdictions requiring all-party consent, all participants must be informed and consent to recording before any recording occurs.
- **User Notification:** Server operators are required to inform all users when voice recording is enabled. This notification must be clear, conspicuous, and provided before recording begins.
- **Data Retention:** Configure `retention_days` to reflect your compliance requirements. Limit retention periods to the minimum necessary for your use case and delete recordings promptly after the retention period expires.
- **Data Access & Security:** Implement appropriate access controls and security measures for recorded data. Handle all recorded data in accordance with GDPR, CCPA, and other applicable privacy regulations.
- **Best Practices:** Review and update both `retention_days` and `save_mode` settings to ensure they align with your legal obligations, privacy policy, and security best practices. Consider using `save_mode: "flagged"` to minimize data collection when possible.

### Production Deployment

1. Use a reverse proxy (nginx/Apache) for SSL termination
2. Configure proper server keys for authentication
3. Set up health checks for container orchestration

## Performance Tuning

- Use Faster Whisper (default) for best performance
- Choose model size: `tiny` (fastest) to `large-v3` (most accurate)
- Adjust `compute_type`: `int8` for CPU (fastest)
- Configure CPU threads based on your cores
- Set `transcription.beam_size: 1` and `processing.worker_count: 2+` in processor `config.json`
- Tune plugin `audio_capture.silence_threshold_ms` (default 500) for faster end-of-utterance detection

## Security

- API key authentication
- Rate limiting
- Input validation
- Secure file handling with automatic cleanup
- CORS configuration

For security vulnerabilities, see SECURITY.md.

### Server-side license enforcement (processor)

The processor verifies each connecting client's license against the PandaScript key
API before doing any work, and again on the `/report/evidence` and `/report/audio`
endpoints. Because this runs on the processor, a patched/cracked plugin jar cannot
bypass it: even if the client-side `LicenseManager` is defeated, a connection with an
invalid, expired, or inactive license is rejected and logged (with IP) in the pool
audit log. Premium report-evidence sync therefore only works for valid licenses.

Configure it under `license_enforcement` in the processor `config.json` (see
`config.example.json`):

- `enabled` — `true`/`false`. Omit the block to default ON for pool processors, OFF for self-hosted.
- `mode` — `enforce` (reject bad licenses) or `log_only` (record offender IPs without disconnecting, for a safe rollout).
- `grace_seconds` — a license verified OK within this window keeps working if the key API is briefly unreachable, so an upstream outage never disconnects paying customers; never-verified licenses are still rejected during an outage.
- `cache_ttl_seconds`, `verify_urls`, `product_id`, `timeout_seconds` — tuning / endpoints.

Clients also send a signed handshake (`client_version` + `client_sig`, an HMAC over the
license/product/version). The processor recomputes it and records `client_integrity`
(`ok` / `mismatch` / `absent`) per connection, so tampered or rebuilt clients are
flagged in the audit log even when they present a valid key.

### Live monitoring dashboard (plugin)

The web dashboard includes a "Live monitoring" tab with three panels:

- **Live transcript stream** — recent buffered transcript lines across every online speaker, newest first, flagged lines highlighted.
- **Flagged-event feed** — recent flagged voice lines plus the unreviewed-report count.
- **Per-player voice history** — one player's recent transcript, processor evidence, and mute history in a single view.

This is a standard part of the plugin, not a separate add-on. The whole plugin requires a valid
license to run at all, and license verification is applied throughout (see the server-side and
integrity sections above), so these panels are available on any licensed install. The richest
data — processor evidence/audio — comes through `/report/evidence` and `/report/audio`, which
verify the license server-side, so the information is only ever served to a verified client.

Each panel can be toggled independently in the plugin `config.yml` under the web dashboard
settings (all default on).

## Monitoring and Logging

### Log Files
- `logs/processor.log` - Backend processing logs

### Endpoints
- `/health` - Health check
- `/stats` - System statistics
- `/report/evidence` - License-scoped transcript evidence (requires `report_buffer.enabled`; authenticate with `X-License-Key`, optional `X-Server-Key` when configured)

## License

This project is licensed under the VoiceSentinel Source-Available License v1.0.
You may view and contribute to the code, but may not use, redistribute, or incorporate it
in your own projects without explicit permission.

Unauthorized use will result in DMCA takedowns and legal action.

See LICENSE file for details.

## Support

- Documentation: [Gitbook](https://pandascript.gitbook.io/pandascript/voicesentinel-processor)
- Discord: [Join our community](https://discord.gg/JAJyuzdgHZ)
- Issues: [GitHub Issues](https://github.com/jackh54/VoiceSentinel/issues)
- Security: security@pandadevv.dev

## Changelog

### Plugin 3.9.0
- On-Server Mode: the speech engine now downloads from BuiltByBit, so a fresh install sets itself up faster and more reliably
- Engine setup never hangs: it gives up cleanly after two minutes and tells you what to do
- Servers that share a plugin folder no longer download the engine twice at once
- You can still place `engine-<platform>.zip` in `plugins/VoiceSentinel/engine/` by hand; VoiceSentinel checks it and installs it
- Clearer console messages when the engine can't be set up
- General performance and stability improvements

### Plugin 3.8.3
- Faster recovery when the connection to the speech processor drops: VoiceSentinel now notices a dead connection within seconds and reconnects on its own
- Fixed `/vs reload` sometimes dropping the processor connection
- Fixed auto-mute stopping for a player for the rest of their session when the mute ladder had no valid tiers
- The `voicesentinel.reload`, `voicesentinel.stats` and `voicesentinel.vcmute` permissions now work on their own, as documented
- `/vs` help now lists every subcommand
- More accurate anonymous usage statistics (turn off with `telemetry.enabled: false`)
- General performance and stability improvements

### Plugin 3.8.2
- Bug and performance fixes

### Plugin 3.8.1 / Engine 1.1.1
- Faster live checks, mostly on older CPUs: a live hit is confirmed by one model instead of two
- End-of-sentence checks no longer wait behind other players' checks
- Default mute broadcast now reads "{player} was voice muted." (only changed if you never edited it)

### Plugin 3.8.0 / Engine 1.1.0
- Live detection in On-Server Mode: a listed word is caught while the player is still talking, about 0.7 s after the word on a modern CPU
- Every live hit is confirmed by Double Check; the end-of-sentence check stays as a backstop
- New default word list: whole-word matching, a filled moderate tier, self-harm phrases and slurs as mute words
- Cleaner console and chat messages, one line per mute or flag

### Plugin 3.5.0
- On-Server Mode: no Docker, no cloud, audio stays on your machine
- Double Check: a second speech model confirms hits in English On-Server Mode
- Consent options: join notice or confirm/deny menu, with optional kick
- 1-day signed license lease, developer API events, `/vs test` and `/vs debug`

### Processor 3.3.1
- Fix race condition in report buffer retention cleanup (GHSA-vhqx-3242-77j7)

Built by pandadevv
