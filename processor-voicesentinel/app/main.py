#!/usr/bin/env python3

import os
import json
import logging
import asyncio
import time
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.transcriber import create_transcriber
from app.profanity import ProfanityFilter
from app.config_validator import validate_config, ConfigValidationError
from app.llm_profanity import LLMProfanityDetector
from app.console_status import ConsoleStatus
from app.pool_audit import setup_pool_audit, pool_audit_emit, pool_transcript_append
from app.report_buffer import (
    report_buffer_append_async,
    query_report_evidence,
    query_report_audio,
    sanitize_player_query,
    sanitize_seconds_seconds,
    check_evidence_rate_limits,
)

import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
logging.getLogger("huggingface_hub").setLevel(logging.INFO)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

VERSION = "3.2.1"


def license_from_auth_payload(data: dict) -> tuple[str, str]:
    license_plain = ""
    lk = data.get("license_key")
    if isinstance(lk, str) and lk.strip():
        license_plain = lk.strip()
    elif lk is not None and lk != "" and not isinstance(lk, str):
        license_plain = str(lk).strip()
    lic_b64 = data.get("license_key_b64") or data.get("licenseKeyB64") or ""
    if not license_plain and lic_b64:
        if not isinstance(lic_b64, str):
            lic_b64 = str(lic_b64)
        try:
            license_plain = base64.b64decode(lic_b64).decode("utf-8").strip()
        except Exception:
            pass
    fingerprint = ""
    if license_plain:
        fingerprint = hashlib.sha256(license_plain.encode("utf-8")).hexdigest()
    return fingerprint, license_plain


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_local_path = os.path.join(project_root, 'config.json')
    config_path = os.environ.get('CONFIG_PATH', default_local_path)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config_clean = {k: v for k, v in config.items() if not k.startswith('_')}
        validate_config(config_clean)
        return config_clean
    except FileNotFoundError:
        default_config = get_default_config()
        validate_config(default_config)
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {config_path}")
        except OSError as e:
            logger.warning(f"Could not write default config to {config_path}: {e}")
        return default_config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

def get_default_config():
    return {
        "server": {"host": "0.0.0.0", "port": 28472, "server_key": ""},
        "transcription": {
            "model": "Systran/faster-whisper-base",
            "language": "en",
            "device": "cpu",
            "compute_type": "int8",
            "timeout_seconds": 30,
            "cpu_threads": 2,
            "huggingface_token": ""
        },
        "audio": {
            "min_audio_length_ms": 50,
            "max_audio_length_ms": 30000,
            "sample_rate": 16000,
            "channels": 1
        },
        "processing": {
            "queue_max_size": 500
        },
        "recordings": {
            "save_mode": "none",
            "save_path": "recordings/",
            "retention_days": 7
        },
        "response": {"include_audio": False},
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
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
            "fallback_on_error": True
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

class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
        self._license_fingerprints = {}
        self._license_plain = {}
        self._auth_server_keys = {}
        self.transcriber = None
        self.llm_detector = None
        self.config = {}
        self.audio_buffers = {}
        self.buffer_metadata = {}
        self.console_status = None
        self._processing_queue = None
        self._worker_task = None
    
    async def initialize(self, config):
        self.config = config
        self.console_status = ConsoleStatus(config)
        queue_max = config.get("processing", {}).get("queue_max_size", 500)
        self._processing_queue = asyncio.Queue(maxsize=max(1, queue_max))
        self._worker_task = asyncio.create_task(self._processing_worker())
        transcription_config = config.get("transcription", {})
        
        self.transcriber = create_transcriber(
            "faster-whisper",
            model_name=transcription_config.get("model", "Systran/faster-whisper-large-v3"),
            language=transcription_config.get("language", "auto"),
            config=config
        )
        
        self.llm_detector = LLMProfanityDetector(config)
        
        recordings_config = config.get("recordings", {})
        save_mode = recordings_config.get("save_mode", "none")
        if save_mode != "none":
            save_path = Path(recordings_config.get("save_path", "recordings/"))
            save_path.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_recordings(save_path, recordings_config.get("retention_days", 7))
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def set_client_license(self, client_id: str, fingerprint: str, license_plain: str) -> None:
        self._license_fingerprints[client_id] = fingerprint or ""
        self._license_plain[client_id] = license_plain or ""

    def set_auth_server_key(self, client_id: str, server_key: str) -> None:
        self._auth_server_keys[client_id] = server_key if isinstance(server_key, str) else ""

    def get_auth_server_key(self, client_id: str) -> str:
        return self._auth_server_keys.get(client_id, "")

    def get_license_fingerprint(self, client_id: str) -> str:
        return self._license_fingerprints.get(client_id, "")

    def get_license_plain(self, client_id: str) -> str:
        return self._license_plain.get(client_id, "")

    def disconnect(self, client_id: str, skip_pool_disconnect_audit: bool = False) -> None:
        if self.config.get("pool_server") and not skip_pool_disconnect_audit:
            fp = self.get_license_fingerprint(client_id)
            lk = self.get_license_plain(client_id)
            pool_audit_emit(self.config, "disconnect", client_id, fp, license_key=lk)
        self._license_fingerprints.pop(client_id, None)
        self._license_plain.pop(client_id, None)
        self._auth_server_keys.pop(client_id, None)
        self.active_connections.pop(client_id, None)
        self.audio_buffers.pop(client_id, None)
        self.buffer_metadata.pop(client_id, None)
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Send failed: {e}")
                self.disconnect(client_id)
    
    def _max_buffer_bytes(self) -> int:
        max_ms = self.config.get("audio", {}).get("max_audio_length_ms", 30000)
        return 44 + int((max_ms * 32))
    
    async def process_audio(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None, language_word_lists: dict = None):
        try:
            self.buffer_metadata[client_id] = {
                'player_name': player_name,
                'session_id': session_id,
                'profanity_words': profanity_words,
                'language_word_lists': language_word_lists or {}
            }
            
            if client_id not in self.audio_buffers:
                self.audio_buffers[client_id] = b''
            buf = self.audio_buffers[client_id]
            max_bytes = self._max_buffer_bytes()
            if len(buf) + len(audio_data) > max_bytes:
                keep = max_bytes - len(audio_data)
                if keep <= 0:
                    buf = audio_data[-max_bytes:]
                else:
                    buf = (buf[-keep:] if len(buf) > keep else buf) + audio_data
                self.audio_buffers[client_id] = buf
            else:
                self.audio_buffers[client_id] = buf + audio_data
        except Exception as e:
            logger.error(f"Audio buffering failed: {e}")
    
    async def process_audio_final(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None, language_word_lists: dict = None):
        try:
            final_audio = self.audio_buffers.get(client_id, b'') + audio_data
            self.audio_buffers[client_id] = b''
            
            if not self._should_process_audio(final_audio):
                await self.send_message(client_id, {
                    "type": "final_transcript",
                    "transcript": "",
                    "flagged": False,
                    "bad_words": [],
                    "player": player_name or client_id,
                    "session_id": session_id or client_id,
                    "processing_time_ms": 0,
                    "detected_language": "unknown"
                })
                return
            job = {
                "client_id": client_id,
                "final_audio": final_audio,
                "profanity_words": profanity_words,
                "player_name": player_name,
                "session_id": session_id,
                "language_word_lists": language_word_lists,
            }
            try:
                self._processing_queue.put_nowait(job)
            except asyncio.QueueFull:
                logger.warning(f"Processing queue full, dropping recording for client {client_id}")
                fp = self.get_license_fingerprint(client_id)
                pool_audit_emit(
                    self.config,
                    "queue_drop",
                    client_id,
                    fp,
                    license_key=self.get_license_plain(client_id),
                    player=player_name,
                    session_id=session_id or client_id,
                )
                await self.send_message(client_id, {
                    "type": "final_transcript",
                    "transcript": "",
                    "flagged": False,
                    "bad_words": [],
                    "player": player_name or client_id,
                    "session_id": session_id or client_id,
                    "processing_time_ms": 0,
                    "detected_language": "unknown"
                })
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
    
    def _cleanup_old_recordings(self, save_path: Path, retention_days: int):
        try:
            cutoff = time.time() - (retention_days * 86400)
            for file in save_path.glob("*.wav"):
                if file.stat().st_mtime < cutoff:
                    file.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup old recordings: {e}")
    
    async def _save_recording(self, audio_data: bytes, player_name: str, flagged: bool, transcript: str):
        try:
            recordings_config = self.config.get("recordings", {})
            save_mode = recordings_config.get("save_mode", "none")
            
            if save_mode == "none" or (save_mode == "flagged" and not flagged):
                return
            
            save_path = Path(recordings_config.get("save_path", "recordings/"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{player_name}_{'flagged' if flagged else 'clean'}.wav"
            
            audio_file = save_path / filename
            audio_file.write_bytes(audio_data)
            
            metadata_file = save_path / f"{filename}.json"
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "player": player_name,
                "flagged": flagged,
                "transcript": transcript
            }
            metadata_file.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
    
    def _should_process_audio(self, audio_data: bytes) -> bool:
        try:
            if len(audio_data) < 1000:
                return False
            audio_config = self.config.get("audio", {})
            min_duration = audio_config.get("min_audio_length_ms", 500)
            max_duration = audio_config.get("max_audio_length_ms", 30000)
            estimated_duration_ms = (len(audio_data) - 44) / 32
            if estimated_duration_ms < min_duration:
                return False
            if estimated_duration_ms > max_duration:
                return False
            return True
        except Exception:
            return False
    
    async def _process_audio_async(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None, language_word_lists: dict = None):
        processing_start_time = time.monotonic()
        self.console_status.increment_processing()
        try:
            if client_id not in self.active_connections:
                return
            
            actual_player_name = player_name or client_id
            
            transcription_timeout = self.config.get("transcription", {}).get("timeout_seconds", 30)
            transcript, detected_language = await asyncio.wait_for(
                self.transcriber.transcribe(audio_data),
                timeout=float(transcription_timeout)
            )
            
            if client_id not in self.active_connections:
                return
            
            processing_time_ms = max(0, int((time.monotonic() - processing_start_time) * 1000))
            
            language_profanity = []
            language_mute = []
            if language_word_lists and detected_language in language_word_lists:
                language_profanity = language_word_lists.get(detected_language, {}).get("PROFANITY", [])
                language_mute = language_word_lists.get(detected_language, {}).get("MUTE", [])
            else:
                language_profanity = profanity_words
            
            profanity_flagged_words = []
            if transcript and language_profanity:
                temp_filter = ProfanityFilter(words=language_profanity, case_sensitive=False, partial_match=True)
                profanity_flagged_words = temp_filter.check_text(transcript)
            
            mute_flagged_words = []
            if transcript and language_mute:
                temp_filter = ProfanityFilter(words=language_mute, case_sensitive=False, partial_match=True)
                mute_flagged_words = temp_filter.check_text(transcript)
            
            is_profane = len(profanity_flagged_words) > 0
            should_mute = len(mute_flagged_words) > 0
            all_flagged_words = profanity_flagged_words + mute_flagged_words
            
            llm_flagged = False
            llm_confidence = 0.0
            llm_reason = ""
            
            if not is_profane and not should_mute and self.llm_detector and self.llm_detector.enabled and transcript:
                try:
                    logger.info(f"[{actual_player_name}] LLM check: Checking transcript (no word list match)")
                    llm_flagged, llm_confidence, llm_reason = await self.llm_detector.check_profanity(transcript, detected_language)
                    logger.info(f"[{actual_player_name}] LLM result: flagged={llm_flagged}, confidence={llm_confidence:.2f}, reason={llm_reason}")
                    if llm_flagged:
                        is_profane = True
                except Exception as e:
                    logger.error(f"LLM check error: {e}")
            
            if is_profane or should_mute:
                self.console_status.increment_flagged(actual_player_name)
                if should_mute:
                    self.console_status.increment_muted(actual_player_name)
            
            self.console_status.add_transcript(actual_player_name, transcript or "", detected_language, is_profane or should_mute, should_mute)
            
            actual_session_id = session_id or client_id
            
            response_data = {
                "type": "final_transcript",
                "transcript": transcript or "",
                "flagged": is_profane or should_mute,
                "should_mute": should_mute,
                "bad_words": all_flagged_words,
                "profanity_words": profanity_flagged_words,
                "mute_words": mute_flagged_words,
                "player": actual_player_name,
                "session_id": actual_session_id,
                "processing_time_ms": max(0, processing_time_ms),
                "detected_language": detected_language
            }

            if llm_flagged:
                response_data["llm_flagged"] = True
                response_data["llm_confidence"] = llm_confidence
                response_data["llm_reason"] = llm_reason

            if (is_profane or should_mute) and self.config.get("response", {}).get("include_audio", False):
                import base64
                response_data["audio_data"] = base64.b64encode(audio_data).decode("utf-8")
            
            await self.send_message(client_id, response_data)
            self.console_status.increment_processed()
            if self.config.get("report_buffer", {}).get("enabled"):
                fp_rb = self.get_license_fingerprint(client_id)
                if fp_rb and (transcript or "").strip():
                    rb_cfg = self.config.get("report_buffer") or {}
                    rec = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "player": actual_player_name,
                        "session_id": actual_session_id,
                        "transcript": transcript or "",
                        "detected_language": detected_language,
                        "flagged": bool(is_profane or should_mute),
                    }
                    wav = audio_data if rb_cfg.get("save_audio") else None
                    sk_auth = self.get_auth_server_key(client_id)
                    await report_buffer_append_async(self.config, fp_rb, sk_auth, rec, wav)
            if self.config.get("pool_server"):
                fp = self.get_license_fingerprint(client_id)
                lk = self.get_license_plain(client_id)
                pool_transcript_append(
                    self.config,
                    fp,
                    lk,
                    {
                        "event": "transcript",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "client_id": client_id,
                        "player": actual_player_name,
                        "session_id": actual_session_id,
                        "transcript": transcript or "",
                        "flagged": is_profane or should_mute,
                        "should_mute": should_mute,
                        "detected_language": detected_language,
                        "processing_time_ms": processing_time_ms,
                    },
                )
                if is_profane or should_mute:
                    pool_audit_emit(
                        self.config,
                        "transcription_flagged",
                        client_id,
                        fp,
                        license_key=lk,
                        player=actual_player_name,
                        session_id=actual_session_id,
                        profanity_hits=len(profanity_flagged_words),
                        mute_hits=len(mute_flagged_words),
                        llm_flagged=bool(llm_flagged),
                    )
            await self._save_recording(audio_data, actual_player_name, is_profane, transcript or "")
            
        except asyncio.TimeoutError:
            logger.error(f"Processing timeout after {transcription_timeout}s")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
        finally:
            self.console_status.decrement_processing()
    
    async def _processing_worker(self):
        while self._processing_queue is not None:
            try:
                job = await self._processing_queue.get()
                client_id = job["client_id"]
                if client_id not in self.active_connections:
                    continue
                await self._process_audio_async(
                    client_id,
                    job["final_audio"],
                    job["profanity_words"],
                    job.get("player_name"),
                    job.get("session_id"),
                    job.get("language_word_lists"),
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing worker error: {e}")

ws_manager = WebSocketManager()

async def console_update_task():
    while True:
        try:
            await asyncio.sleep(5)
            if ws_manager.console_status and ws_manager.console_status.has_changed():
                ws_manager.console_status.print_status()
        except Exception as e:
            logger.error(f"Console update error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_pool_audit(
        bool(config.get("pool_server", False)),
        config.get("pool_server_audit_log", "logs/pooled_server_audit.jsonl"),
    )
    if config.get("pool_server"):
        Path(config.get("pool_server_transcripts_dir", "logs/pool_transcripts_by_license")).mkdir(
            parents=True, exist_ok=True
        )
    rb_path = (config.get("report_buffer") or {}).get("path", "report_buffer/")
    if rb_path:
        Path(rb_path).mkdir(parents=True, exist_ok=True)
    await ws_manager.initialize(config)
    if ws_manager.console_status:
        ws_manager.console_status.print_status(force=True)
    asyncio.create_task(console_update_task())
    yield
    if ws_manager._worker_task is not None:
        ws_manager._worker_task.cancel()
        try:
            await ws_manager._worker_task
        except asyncio.CancelledError:
            pass

config = load_config()

app = FastAPI(
    title="VoiceSentinel Processor",
    description="Real-time audio processing and transcription service",
    version=VERSION,
    lifespan=lifespan
)


cors_config = config.get("cors", {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get("allow_origins", ["*"]),
    allow_credentials=cors_config.get("allow_credentials", True),
    allow_methods=cors_config.get("allow_methods", ["*"]),
    allow_headers=cors_config.get("allow_headers", ["*"]),
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": VERSION}

@app.get("/report/evidence")
async def report_evidence(
    request: Request,
    player: str = Query(..., min_length=1, max_length=64),
    seconds: int = Query(300, ge=30, le=86400),
):
    rb = config.get("report_buffer") or {}
    if not rb.get("enabled"):
        raise HTTPException(
            status_code=404,
            detail="Report buffer disabled (set report_buffer.enabled true in processor config)",
        )
    license_key = request.headers.get("x-license-key") or ""
    if not license_key:
        auth = request.headers.get("authorization") or ""
        if isinstance(auth, str) and auth.lower().startswith("bearer "):
            license_key = auth[7:].strip()
    if not license_key.strip():
        raise HTTPException(status_code=403, detail="Missing license key")
    fp_ev, _plain = license_from_auth_payload({"license_key": license_key.strip()})
    if not fp_ev:
        raise HTTPException(status_code=403, detail="Invalid license key")
    client_ip = ""
    try:
        if request.client and request.client.host:
            client_ip = request.client.host
    except Exception:
        client_ip = ""
    if not await check_evidence_rate_limits(fp_ev, client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    cfg_key = (config.get("server") or {}).get("server_key") or ""
    sk_for_query = ""
    if isinstance(cfg_key, str) and cfg_key.strip():
        req_key = request.headers.get("x-server-key") or ""
        if req_key != cfg_key:
            raise HTTPException(status_code=401, detail="Invalid server key")
        sk_for_query = req_key
    else:
        sk_for_query = request.headers.get("x-server-key") or ""
    pq = sanitize_player_query(player)
    if not pq:
        raise HTTPException(status_code=400, detail="Invalid player parameter")
    sec = sanitize_seconds_seconds(seconds)
    events = query_report_evidence(config, fp_ev, sk_for_query, pq, sec)
    return {"events": events}


@app.get("/report/audio")
async def report_audio(
    request: Request,
    player: str = Query(..., min_length=1, max_length=64),
    session_id: str = Query(..., min_length=1, max_length=128),
):
    from fastapi.responses import Response as RawResponse

    rb = config.get("report_buffer") or {}
    if not rb.get("enabled") or not rb.get("save_audio"):
        raise HTTPException(status_code=404, detail="Audio storage disabled")
    license_key = request.headers.get("x-license-key") or ""
    if not license_key:
        auth = request.headers.get("authorization") or ""
        if isinstance(auth, str) and auth.lower().startswith("bearer "):
            license_key = auth[7:].strip()
    if not license_key.strip():
        raise HTTPException(status_code=403, detail="Missing license key")
    fp_ev, _plain = license_from_auth_payload({"license_key": license_key.strip()})
    if not fp_ev:
        raise HTTPException(status_code=403, detail="Invalid license key")
    client_ip = ""
    try:
        if request.client and request.client.host:
            client_ip = request.client.host
    except Exception:
        client_ip = ""
    if not await check_evidence_rate_limits(fp_ev, client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    sk_for_query = request.headers.get("x-server-key") or ""
    pq = sanitize_player_query(player)
    if not pq:
        raise HTTPException(status_code=400, detail="Invalid player parameter")
    wav = query_report_audio(config, fp_ev, sk_for_query, pq, session_id)
    if not wav:
        raise HTTPException(status_code=404, detail="Audio not found")
    return RawResponse(content=wav, media_type="audio/wav")


@app.get("/stats")
async def get_stats():
    qsize = ws_manager._processing_queue.qsize() if ws_manager._processing_queue else 0
    queue_max = config.get("processing", {}).get("queue_max_size", 500)
    try:
        queue_max = max(1, int(queue_max))
    except (TypeError, ValueError):
        queue_max = 500
    queue_utilization = round(100.0 * min(qsize, queue_max) / queue_max, 2)
    return {
        "active_connections": len(ws_manager.active_connections),
        "transcriber_ready": ws_manager.transcriber is not None,
        "processing_queue_size": qsize,
        "queue_max_size": queue_max,
        "queue_utilization_percent": queue_utilization,
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    authenticated = False
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            if message_type == "auth":
                server_key = data.get("server_key", "")
                config_key = config.get("server", {}).get("server_key", "")

                fingerprint, license_plain = license_from_auth_payload(data)

                remote_ip = ""
                try:
                    if websocket.client and websocket.client.host:
                        remote_ip = websocket.client.host
                except Exception:
                    remote_ip = ""

                if not config_key or server_key == config_key:
                    authenticated = True
                    ws_manager.set_client_license(client_id, fingerprint, license_plain)
                    ws_manager.set_auth_server_key(
                        client_id, server_key if isinstance(server_key, str) else ""
                    )
                    pool_audit_emit(
                        config,
                        "auth_success",
                        client_id,
                        fingerprint,
                        license_key=license_plain,
                        remote_ip=remote_ip,
                    )
                    await ws_manager.send_message(client_id, {"type": "auth_success"})
                else:
                    pool_audit_emit(
                        config,
                        "auth_failed",
                        client_id,
                        fingerprint,
                        license_key=license_plain,
                        remote_ip=remote_ip,
                        reason="invalid_server_key",
                    )
                    await ws_manager.send_message(client_id, {"type": "auth_failed", "reason": "Invalid server key"})
                    await websocket.close()
                    ws_manager.disconnect(client_id, skip_pool_disconnect_audit=True)
                    return
                    
            elif message_type == "audio_chunk" and authenticated:
                import base64
                audio_b64 = data.get("audio_data", "")
                profanity_words = data.get("profanity_words", [])
                player_name = data.get("player", client_id)
                session_id = data.get("session_id", client_id)
                is_final = data.get("is_final", False)
                language_word_lists = data.get("language_word_lists", {})
                
                if isinstance(profanity_words, str):
                    try:
                        profanity_words = json.loads(profanity_words)
                    except Exception:
                        profanity_words = []

                if isinstance(language_word_lists, str):
                    try:
                        language_word_lists = json.loads(language_word_lists)
                    except Exception:
                        language_word_lists = {}
                
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        if is_final:
                            await ws_manager.process_audio_final(client_id, audio_bytes, profanity_words, player_name, session_id, language_word_lists)
                        else:
                            await ws_manager.process_audio(client_id, audio_bytes, profanity_words, player_name, session_id, language_word_lists)
                    except Exception as e:
                        logger.error(f"Audio decode failed: {e}")
                        
            elif message_type == "ping":
                await ws_manager.send_message(client_id, {"type": "pong", "timestamp": data.get("timestamp", 0)})
            
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        ws_manager.disconnect(client_id)

if __name__ == "__main__":
    config = load_config()
    server_config = config.get("server", {})
    uvicorn.run(
        "main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 28472),
        log_level="info",
        access_log=False
    )