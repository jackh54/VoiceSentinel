# VoiceSentinel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Java](https://img.shields.io/badge/Java-17orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Minecraft](https://img.shields.io/badge/Minecraft-1.21+-green.svg)](https://www.minecraft.net/)

VoiceSentinel is a comprehensive real-time voice moderation system for Minecraft servers using Simple Voice Chat. It combines a Java-based Minecraft plugin with a Python backend processor to provide advanced speech-to-text transcription and content filtering capabilities.

## 🌟 Features

### Core Functionality
- **Real-time voice transcription** using OpenAI Whisper models
- **Advanced profanity detection** with customizable word lists and pattern matching
- **Multi-language support** for international servers
- **Configurable audio processing** with chunk size, sample rate, and duration limits
- **Rate limiting and API key authentication** for secure integration
- **Queue-based processing** for handling high-volume voice chat
- **Comprehensive logging and monitoring** for server administration

### Technical Features
- **Asynchronous processing** with background workers
- **Docker support** for easy deployment
- **CPU-optimized** for cost-effective hosting
- **REST API** for integration with other services
- **File-based result delivery** for reliable communication
- **Automatic cleanup** of temporary files and processed results
- **Health monitoring** and system statistics

### Minecraft Integration
- **Folia & Paper compatibility** for modern server software
- **Simple Voice Chat integration** for seamless voice capture
- **Configurable moderation actions** (warnings, kicks, bans)
- **Player-specific settings** and permissions
- **Server-wide and per-world configuration**

## 📋 Requirements

### Server Requirements
- **Java 17+** for the Minecraft plugin
- **Python 3.8+** for the backend processor
- **Minecraft 1.21+** Folia or Paper
- **Simple Voice Chat** plugin installed
- **Minimum 4GB RAM** (8GB+ recommended)
- **Multi-core CPU** for optimal processing

### Dependencies
- **Backend Processor**: FastAPI, Whisper, PyTorch, NumPy, SciPy

> **Using Docker is recommended to run the backend processor, as it installs this for you.**

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/jackh54/VoiceSentinel.git
cd VoiceSentinel/processor-voicesentinel

# Copy and edit the configuration file
cp config.example.json config.json
# Edit config.json with your settings

# Start the backend processor in the background using Docker Compose
docker compose up -d

# To view logs or check status:
docker compose logs -f
docker compose ps

# To stop the processor:
docker compose down
```

### 2. Plugin Installation

Install and configure the plugin as described in the plugin's documentation.

### 3. Configuration

Configure the backend using `config.json` (see below). For plugin configuration, refer to the plugin's own documentation.

#### Sample Backend Configuration (`config.json`)
```json
{
  "transcriber": {
    "type": "whisper",
    "model_name": "tiny.en",
    "language": "en"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "processing": {
    "max_concurrent_jobs": 4,
    "queue_warning_threshold": 200
  },
  "audio": {
    "max_chunk_size": 262144,
    "max_total_size": 5242880,
    "sample_rate": 16000
  }
}
```

### Production Deployment

1. **Use a reverse proxy** (nginx/Apache) for SSL termination
2. **Implement backup strategies** for configuration and logs

### Performance Tuning

- **Adjust worker count** based on CPU cores
- **Optimize queue sizes** for your server load
- **Configure memory limits** for audio processing
- **Use appropriate Whisper models** (tiny/base/small/medium/large)
- **Monitor processing times** and adjust timeouts

## 🛡️ Security

VoiceSentinel includes comprehensive security features:

- **API key authentication** for all endpoints
- **Rate limiting** to prevent abuse
- **Input validation** and sanitization
- **Secure file handling** with automatic cleanup
- **CORS configuration** for web integration
- **Trusted host middleware** for production deployments

For security vulnerabilities, please see our [Security Policy](SECURITY.md).

## 📈 Monitoring and Logging

### Log Files
- `logs/processor.log` - Backend processing logs

### Metrics
- Processing queue length
- Average processing time
- Error rates and types
- Memory and CPU usage
- Audio processing statistics

### Health Checks
- `/health` endpoint for monitoring
- Automatic worker health monitoring
- Queue overflow detection
- Resource usage alerts

## 🤝 Contributing

I welcome contributions! Please see the contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Backend development
cd processor-voicesentinel
pip install -r requirements.txt
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: soon
- **Discord**: [Join our community](https://discord.gg/JAJyuzdgHZ)
- **Issues**: [GitHub Issues](https://github.com/jackh54/VoiceSentinel/issues)
- **Security**:security@pandadevv.dev](mailto:security@pandadevv.dev)

## 🗺️ Roadmap

### Roadmap (v1.0.0+)

- [ ] Improved voice data transmission to API
- [ ] Basic multi-language support
- [ ] Improved profanity detection
- [ ] Web dashboard (maybe?)
- [ ] Player reputation system (initial version)
- [ ] Proper Pterodactyl Egg setup (current egg shouldn't be used for production)

### Technical Improvements
- [ ] Optimize performance for large servers
- [ ] Microservices for scalability
- [ ] Add GraphQL API
- [ ] WebSocket support for real-time updates
- [ ] Improve caching
- [ ] Kubernetes support

---

*Built with ❤️ by pandadevv*
