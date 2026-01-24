# VoiceSentinel

[![Java](https://img.shields.io/badge/Java-17-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Minecraft](https://img.shields.io/badge/Minecraft-1.21+-green.svg)](https://www.minecraft.net/)

VoiceSentinel is a real-time voice moderation system for Minecraft servers using Simple Voice Chat. It combines a Java plugin with a Python processor for speech-to-text transcription and content filtering.

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
- Simple Voice Chat integration

## Requirements

### Server Requirements
- Java 17+ for the Minecraft plugin
- Python 3.8+ for the processor
- Minecraft 1.21+ Folia or Paper
- Simple Voice Chat plugin
- Minimum 4GB RAM (8GB+ recommended)
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

## Security

- API key authentication
- Rate limiting
- Input validation
- Secure file handling with automatic cleanup
- CORS configuration

For security vulnerabilities, see SECURITY.md.

## Monitoring and Logging

### Log Files
- `logs/processor.log` - Backend processing logs

### Endpoints
- `/health` - Health check
- `/stats` - System statistics

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

## Roadmap

### Version 3.0.0 (Current)
- Migrated to Faster Whisper large-v3 model
- Multi-language support with auto-detection
- Recording save functionality
- Simplified configuration
- Improved performance

### Future Plans
- Audio clip attachments for Discord webhooks
- Web dashboard
- Player reputation system
- Microservices architecture for scalability

Built by pandadevv
