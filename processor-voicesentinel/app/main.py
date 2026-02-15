#!/usr/bin/env python3

import os
import json
import logging
import asyncio
import time
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.transcriber import create_transcriber
from app.profanity import ProfanityFilter
from app.config_validator import validate_config, ConfigValidationError
from app.llm_profanity import LLMProfanityDetector
from app.console_status import ConsoleStatus

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

VERSION = "3.0.0"

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
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }
    }

class WebSocketManager:
    def __init__(self):
        self.active_connections = {}
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
    
    def disconnect(self, client_id: str):
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
                "processing_time_ms": processing_time_ms,
                "detected_language": detected_language
            }
            
            if llm_flagged:
                response_data["llm_flagged"] = True
                response_data["llm_confidence"] = llm_confidence
                response_data["llm_reason"] = llm_reason
            
            await self.send_message(client_id, response_data)
            self.console_status.increment_processed()
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

@app.get("/stats")
async def get_stats():
    qsize = ws_manager._processing_queue.qsize() if ws_manager._processing_queue else 0
    return {
        "active_connections": len(ws_manager.active_connections),
        "transcriber_ready": ws_manager.transcriber is not None,
        "processing_queue_size": qsize
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
                
                if not config_key or server_key == config_key:
                    authenticated = True
                    await ws_manager.send_message(client_id, {"type": "auth_success"})
                else:
                    await ws_manager.send_message(client_id, {"type": "auth_failed", "reason": "Invalid server key"})
                    await websocket.close()
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
                    except:
                        profanity_words = []
                
                if isinstance(language_word_lists, str):
                    try:
                        language_word_lists = json.loads(language_word_lists)
                    except:
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