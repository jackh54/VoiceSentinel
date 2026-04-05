import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    pass

class ConfigValidator:
    def __init__(self):
        self.errors: List[str] = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        self.errors = []
        self._validate_server(config)
        self._validate_pool_server(config)
        self._validate_transcription(config)
        self._validate_audio(config)
        self._validate_processing(config)
        self._validate_report_buffer(config)
        return len(self.errors) == 0, self.errors, []
    
    def _validate_server(self, config: Dict[str, Any]):
        server = config.get("server", {})
        if not server:
            self.errors.append("Missing server config")
            return
        port = server.get("port", 28472)
        if not isinstance(port, int) or port < 1 or port > 65535:
            self.errors.append(f"Invalid port: {port}")

    def _validate_pool_server(self, config: Dict[str, Any]):
        if "pool_server" not in config:
            return
        ps = config.get("pool_server")
        if not isinstance(ps, bool):
            self.errors.append(f"pool_server must be a boolean, got: {type(ps).__name__}")
        log_path = config.get("pool_server_audit_log")
        if log_path is not None and not isinstance(log_path, str):
            self.errors.append(f"pool_server_audit_log must be a string, got: {type(log_path).__name__}")
        tx_dir = config.get("pool_server_transcripts_dir")
        if tx_dir is not None and not isinstance(tx_dir, str):
            self.errors.append(f"pool_server_transcripts_dir must be a string, got: {type(tx_dir).__name__}")

    def _validate_transcription(self, config: Dict[str, Any]):
        transcription = config.get("transcription", {})
        if not transcription:
            self.errors.append("Missing transcription config")
            return
        timeout = transcription.get("timeout_seconds", 30)
        if not isinstance(timeout, (int, float)) or timeout < 1:
            self.errors.append(f"Invalid timeout: {timeout}")
    
    def _validate_audio(self, config: Dict[str, Any]):
        audio = config.get("audio", {})
        if not audio:
            return
        
        min_audio_length_ms = audio.get("min_audio_length_ms", 50)
        max_audio_length_ms = audio.get("max_audio_length_ms", 30000)
        
        min_valid = True
        max_valid = True
        
        try:
            min_audio_length_ms = int(min_audio_length_ms)
            if min_audio_length_ms <= 0:
                self.errors.append(f"min_audio_length_ms must be a positive integer, got: {audio.get('min_audio_length_ms')}")
                min_valid = False
        except (ValueError, TypeError):
            self.errors.append(f"min_audio_length_ms must be an integer, got: {type(audio.get('min_audio_length_ms')).__name__}")
            min_valid = False
        
        try:
            max_audio_length_ms = int(max_audio_length_ms)
            if max_audio_length_ms <= 0:
                self.errors.append(f"max_audio_length_ms must be a positive integer, got: {audio.get('max_audio_length_ms')}")
                max_valid = False
        except (ValueError, TypeError):
            self.errors.append(f"max_audio_length_ms must be an integer, got: {type(audio.get('max_audio_length_ms')).__name__}")
            max_valid = False
        
        if min_valid and max_valid and min_audio_length_ms >= max_audio_length_ms:
            self.errors.append(f"min_audio_length_ms must be less than max_audio_length_ms")
    
    def _validate_processing(self, config: Dict[str, Any]):
        processing = config.get("processing", {})
        if not processing:
            return
        qmax = processing.get("queue_max_size", 500)
        try:
            qmax = int(qmax)
            if qmax < 1:
                self.errors.append("processing.queue_max_size must be at least 1")
        except (ValueError, TypeError):
            self.errors.append(f"processing.queue_max_size must be an integer, got: {type(qmax).__name__}")

    def _validate_report_buffer(self, config: Dict[str, Any]):
        rb = config.get("report_buffer")
        if rb is None:
            return
        if not isinstance(rb, dict):
            self.errors.append(f"report_buffer must be an object, got: {type(rb).__name__}")
            return
        if "enabled" in rb and not isinstance(rb.get("enabled"), bool):
            self.errors.append("report_buffer.enabled must be a boolean")
        if "save_audio" in rb and not isinstance(rb.get("save_audio"), bool):
            self.errors.append("report_buffer.save_audio must be a boolean")
        path = rb.get("path")
        if path is not None and not isinstance(path, str):
            self.errors.append(f"report_buffer.path must be a string, got: {type(path).__name__}")
        rs_raw = rb.get("retention_seconds", 604800)
        try:
            rs = int(rs_raw)
            if rs < 60 or rs > 31536000:
                self.errors.append("report_buffer.retention_seconds must be between 60 and 31536000")
        except (TypeError, ValueError):
            self.errors.append(
                f"report_buffer.retention_seconds must be an integer, got: {type(rs_raw).__name__}"
            )

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    validator = ConfigValidator()
    is_valid, errors, _ = validator.validate(config)
    if not is_valid:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        raise ConfigValidationError(f"Configuration errors:\n{error_msg}")
    return config
