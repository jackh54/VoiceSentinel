from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config_validator import validate_config, ConfigValidationError

logger = logging.getLogger(__name__)

_ENV_MAPPINGS: Dict[str, Tuple[str, ...]] = {
    "SERVER_KEY": ("server", "server_key"),
    "SERVER_HOST": ("server", "host"),
    "SERVER_PORT": ("server", "port"),
    "POOL_SERVER": ("pool_server",),
    "POOL_SERVER_AUDIT_LOG": ("pool_server_audit_log",),
    "POOL_SERVER_TRANSCRIPTS_DIR": ("pool_server_transcripts_dir",),
    "TRANSCRIPTION_MODEL": ("transcription", "model"),
    "TRANSCRIPTION_LANGUAGE": ("transcription", "language"),
    "TRANSCRIPTION_DEVICE": ("transcription", "device"),
    "TRANSCRIPTION_COMPUTE_TYPE": ("transcription", "compute_type"),
    "TRANSCRIPTION_TIMEOUT_SECONDS": ("transcription", "timeout_seconds"),
    "TRANSCRIPTION_CPU_THREADS": ("transcription", "cpu_threads"),
    "TRANSCRIPTION_BEAM_SIZE": ("transcription", "beam_size"),
    "TRANSCRIPTION_VAD_FILTER": ("transcription", "vad_filter"),
    "TRANSCRIPTION_USE_MEMORY_INPUT": ("transcription", "use_memory_input"),
    "TRANSCRIPTION_HUGGINGFACE_TOKEN": ("transcription", "huggingface_token"),
    "AUDIO_MIN_AUDIO_LENGTH_MS": ("audio", "min_audio_length_ms"),
    "AUDIO_MAX_AUDIO_LENGTH_MS": ("audio", "max_audio_length_ms"),
    "AUDIO_SAMPLE_RATE": ("audio", "sample_rate"),
    "AUDIO_CHANNELS": ("audio", "channels"),
    "PROCESSING_QUEUE_MAX_SIZE": ("processing", "queue_max_size"),
    "PROCESSING_WORKER_COUNT": ("processing", "worker_count"),
    "PROCESSING_STT_CONCURRENCY": ("processing", "stt_concurrency"),
    "RECORDINGS_SAVE_MODE": ("recordings", "save_mode"),
    "RECORDINGS_SAVE_PATH": ("recordings", "save_path"),
    "RECORDINGS_RETENTION_DAYS": ("recordings", "retention_days"),
    "RESPONSE_INCLUDE_AUDIO": ("response", "include_audio"),
    "CORS_ALLOW_ORIGINS": ("cors", "allow_origins"),
    "CORS_ALLOW_CREDENTIALS": ("cors", "allow_credentials"),
    "CORS_ALLOW_METHODS": ("cors", "allow_methods"),
    "CORS_ALLOW_HEADERS": ("cors", "allow_headers"),
    "LLM_PROFANITY_ENABLED": ("llm_profanity", "enabled"),
    "LLM_PROFANITY_PROVIDER": ("llm_profanity", "provider"),
    "LLM_PROFANITY_API_KEY": ("llm_profanity", "api_key"),
    "LLM_PROFANITY_MODEL": ("llm_profanity", "model"),
    "LLM_PROFANITY_TIMEOUT_SECONDS": ("llm_profanity", "timeout_seconds"),
    "LLM_PROFANITY_CONFIDENCE_THRESHOLD": ("llm_profanity", "confidence_threshold"),
    "LLM_PROFANITY_STRICTNESS": ("llm_profanity", "strictness"),
    "LLM_PROFANITY_MAX_CONCURRENT_REQUESTS": ("llm_profanity", "max_concurrent_requests"),
    "LLM_PROFANITY_FALLBACK_ON_ERROR": ("llm_profanity", "fallback_on_error"),
    "CONSOLE_LOG_TRANSCRIPTS": ("console", "log_transcripts"),
    "CONSOLE_LIVE_DISPLAY": ("console", "live_display"),
    "REPORT_BUFFER_ENABLED": ("report_buffer", "enabled"),
    "REPORT_BUFFER_PATH": ("report_buffer", "path"),
    "REPORT_BUFFER_RETENTION_SECONDS": ("report_buffer", "retention_seconds"),
    "REPORT_BUFFER_SAVE_AUDIO": ("report_buffer", "save_audio"),
}

_BOOL_KEYS = {
    "POOL_SERVER",
    "TRANSCRIPTION_VAD_FILTER",
    "TRANSCRIPTION_USE_MEMORY_INPUT",
    "RESPONSE_INCLUDE_AUDIO",
    "CORS_ALLOW_CREDENTIALS",
    "LLM_PROFANITY_ENABLED",
    "LLM_PROFANITY_FALLBACK_ON_ERROR",
    "CONSOLE_LOG_TRANSCRIPTS",
    "CONSOLE_LIVE_DISPLAY",
    "REPORT_BUFFER_ENABLED",
    "REPORT_BUFFER_SAVE_AUDIO",
}

_INT_KEYS = {
    "SERVER_PORT",
    "TRANSCRIPTION_TIMEOUT_SECONDS",
    "TRANSCRIPTION_CPU_THREADS",
    "TRANSCRIPTION_BEAM_SIZE",
    "AUDIO_MIN_AUDIO_LENGTH_MS",
    "AUDIO_MAX_AUDIO_LENGTH_MS",
    "AUDIO_SAMPLE_RATE",
    "AUDIO_CHANNELS",
    "PROCESSING_QUEUE_MAX_SIZE",
    "PROCESSING_WORKER_COUNT",
    "PROCESSING_STT_CONCURRENCY",
    "RECORDINGS_RETENTION_DAYS",
    "LLM_PROFANITY_TIMEOUT_SECONDS",
    "LLM_PROFANITY_MAX_CONCURRENT_REQUESTS",
    "REPORT_BUFFER_RETENTION_SECONDS",
}

_FLOAT_KEYS = {
    "LLM_PROFANITY_CONFIDENCE_THRESHOLD",
}

_JSON_KEYS = {
    "CORS_ALLOW_ORIGINS",
    "CORS_ALLOW_METHODS",
    "CORS_ALLOW_HEADERS",
}


def get_default_config() -> Dict[str, Any]:
    return {
        "server": {"host": "0.0.0.0", "port": 28472, "server_key": ""},
        "transcription": {
            "model": "Systran/faster-whisper-base",
            "language": "en",
            "device": "cpu",
            "compute_type": "int8",
            "timeout_seconds": 30,
            "cpu_threads": 2,
            "beam_size": 5,
            "vad_filter": True,
            "use_memory_input": False,
            "huggingface_token": "",
        },
        "audio": {
            "min_audio_length_ms": 50,
            "max_audio_length_ms": 30000,
            "sample_rate": 16000,
            "channels": 1,
        },
        "processing": {
            "queue_max_size": 500,
            "worker_count": 4,
            "stt_concurrency": 4,
        },
        "recordings": {
            "save_mode": "none",
            "save_path": "recordings/",
            "retention_days": 7,
        },
        "response": {"include_audio": False},
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        },
        "llm_profanity": {
            "enabled": False,
            "provider": "ollama",
            "api_key": "http://localhost:11434",
            "model": "llama2",
            "timeout_seconds": 15,
            "confidence_threshold": 0.7,
            "strictness": "medium",
            "max_concurrent_requests": 3,
            "fallback_on_error": True,
        },
        "console": {"log_transcripts": False, "live_display": True},
        "pool_server": False,
        "pool_server_audit_log": "logs/pooled_server_audit.jsonl",
        "pool_server_transcripts_dir": "logs/pool_transcripts_by_license",
        "report_buffer": {
            "enabled": False,
            "path": "report_buffer/",
            "retention_seconds": 604800,
            "save_audio": False,
        },
    }


def _parse_env_value(key: str, raw: str) -> Any:
    if key in _BOOL_KEYS:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        return int(raw)
    if key in _FLOAT_KEYS:
        return float(raw)
    if key in _JSON_KEYS:
        return json.loads(raw)
    return raw


def _set_nested(config: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    node = config
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def _load_env_overrides() -> Tuple[Dict[str, Any], List[str]]:
    overrides: Dict[str, Any] = {}
    applied: List[str] = []
    for env_key, path in _ENV_MAPPINGS.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            value = _parse_env_value(env_key, raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid environment variable %s: %s", env_key, exc)
            continue
        _set_nested(overrides, path, value)
        applied.append(env_key)
    return overrides, applied


def _deep_merge(base: Dict[str, Any], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overlay:
        return base
    result = deepcopy(base)
    for key, value in overlay.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _default_config_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "config.json")


def load_config() -> Dict[str, Any]:
    config_path = os.environ.get("CONFIG_PATH", _default_config_path())
    defaults = get_default_config()
    file_config: Optional[Dict[str, Any]] = None
    used_config_file = False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
        used_config_file = True
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config file: %s", e)
        raise

    env_overrides, env_keys = _load_env_overrides()

    config = _deep_merge(defaults, file_config)
    if file_config:
        config = {k: v for k, v in config.items() if not k.startswith("_")}
    config = _deep_merge(config, env_overrides)

    if not used_config_file and not env_keys:
        validate_config(deepcopy(config))
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=2)
            logger.info("Created default config at %s", config_path)
        except OSError as e:
            logger.warning("Could not write default config to %s: %s", config_path, e)
        logger.info("Config source: built-in defaults (wrote %s)", config_path)
        return defaults

    try:
        validate_config(config)
    except ConfigValidationError as e:
        logger.error("Configuration validation failed: %s", e)
        raise

    if env_keys and used_config_file:
        logger.info(
            "Config source: config.json (%s) with environment overrides: %s",
            config_path,
            ", ".join(sorted(env_keys)),
        )
    elif env_keys:
        logger.info("Config source: environment variables (%s)", ", ".join(sorted(env_keys)))
    else:
        logger.info("Config source: config.json (%s)", config_path)

    return config
