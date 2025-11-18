#!/usr/bin/env python3

import os
import sys
import json
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.transcriber import WhisperTranscriber
from app.profanity import create_default_filter, ProfanityFilter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app_state = {}

def load_config():
    config_path = os.environ.get('CONFIG_PATH', '/app/config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return get_default_config()

def get_default_config():
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 28472
        },
        "transcription": {
            "engine": "whisper",
            "model": "tiny.en",
            "language": "en",
            "timeout_seconds": 30
        },
        "audio": {
            "max_file_size": 5 * 1024 * 1024,  # 5MB
            "supported_formats": ["wav", "mp3", "ogg", "flac"]
        },
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        },
        "security": {
            "api_keys": []
        }
    }

class SimpleWebSocketManager:  
    def __init__(self):
        self.active_connections = {}
        self.transcriber = None
        self.profanity_filter = None
        
        # Voice Activity Detection system
        self.audio_buffers = {}
        self.recording_sessions = {}
        self.recording_timers = {}
        self.buffer_metadata = {}
        
        # VAD configuration
        self.min_talking_time_ms = 500
        self.max_recording_duration_seconds = 30
        self.end_threshold_packets = 5
    
    async def initialize(self, config):
        try:
            transcription_config = config.get("transcription", {})
            self.transcriber = WhisperTranscriber(
                model_name=transcription_config.get("model", "tiny.en"),
                language=transcription_config.get("language", "en"),
                config=config
            )
            self.profanity_filter = create_default_filter()
            
            # Load VAD configuration
            voice_detection = config.get("voice_detection", {})
            self.min_talking_time_ms = voice_detection.get("min_talking_time_ms", 500)
            self.max_recording_duration_seconds = voice_detection.get("max_recording_duration_seconds", 30)
            self.end_threshold_packets = voice_detection.get("end_threshold_packets", 5)
            
            logger.info(f"WebSocket manager initialized with VAD: min_talking={self.min_talking_time_ms}ms, max_duration={self.max_recording_duration_seconds}s")
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.recording_sessions:
            del self.recording_sessions[client_id]
        if client_id in self.recording_timers:
            self.recording_timers[client_id].cancel()
            del self.recording_timers[client_id]
        if client_id in self.buffer_metadata:
            del self.buffer_metadata[client_id]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                if websocket.client_state.name != 'CONNECTED':
                    logger.warning(f"WebSocket for {client_id} is not connected, removing from active connections")
                    self.disconnect(client_id)
                    return
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
        else:
            logger.debug(f"Client {client_id} not in active connections, skipping message")
    
    async def process_audio(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None):
        """Process audio with Voice Activity Detection"""
        try:
            self.buffer_metadata[client_id] = {
                'player_name': player_name,
                'session_id': session_id,
                'profanity_words': profanity_words
            }
            
            if client_id not in self.recording_sessions:
                self.recording_sessions[client_id] = {
                    'is_recording': False,
                    'start_time': None,
                    'last_audio_time': None,
                    'silence_packets': 0,
                    'total_duration': 0
                }
                self.audio_buffers[client_id] = b''
            
            session = self.recording_sessions[client_id]
            current_time = time.time()
            self.audio_buffers[client_id] += audio_data
            session['last_audio_time'] = current_time
            audio_duration = self._estimate_audio_duration(audio_data)
            session['total_duration'] += audio_duration
            
            if not session['is_recording']:
                if session['total_duration'] >= self.min_talking_time_ms:
                    logger.info(f"Voice detected for {client_id}, starting recording (duration: {session['total_duration']:.1f}ms)")
                    await self._start_recording(client_id)
                else:
                    logger.debug(f"Building audio for {client_id}: {session['total_duration']:.1f}ms/{self.min_talking_time_ms}ms")
            else:
                await self._check_recording_end(client_id)
            
        except Exception as e:
            logger.error(f"Audio processing failed for {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "status": "error"
            })
    
    async def process_audio_final(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None):
        """Process final audio chunk - end recording and process immediately"""
        try:
            if client_id in self.recording_sessions:
                await self._end_recording(client_id, force=True)
            
            final_audio = None
            if client_id in self.audio_buffers and self.audio_buffers[client_id]:
                final_audio = self.audio_buffers[client_id] + audio_data
                self.audio_buffers[client_id] = b''
            else:
                final_audio = audio_data
            
            # pre-validate audio before processing
            if self._should_process_audio(final_audio):
                await self._process_audio_async(client_id, final_audio, profanity_words, player_name, session_id)
            else:
                logger.debug(f"Skipping audio processing for {client_id}: audio too short or invalid")
                # send empty result for very short audio
                actual_player_name = player_name if player_name else client_id
                actual_session_id = session_id if session_id else client_id
                await self.send_message(client_id, {
                    "type": "final_transcript",
                    "transcript": "",
                    "flagged": False,
                    "bad_words": [],
                    "player": actual_player_name,
                    "session_id": actual_session_id
                })
                logger.info(f"Audio processing completed for {actual_player_name}: '' (flagged: False)")
            
        except Exception as e:
            logger.error(f"Final audio processing failed for {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "status": "error"
            })
    
    async def _start_recording(self, client_id: str):
        """Start a new recording session"""
        session = self.recording_sessions[client_id]
        session['is_recording'] = True
        session['start_time'] = time.time()
        session['silence_packets'] = 0
        import asyncio
        self.recording_timers[client_id] = asyncio.create_task(
            self._max_duration_timeout(client_id)
        )
        
        logger.info(f"Started recording for {client_id}")
    
    async def _check_recording_end(self, client_id: str):
        """Check if recording should end based on silence or duration"""
        session = self.recording_sessions[client_id]
        current_time = time.time()
        if session['last_audio_time'] and (current_time - session['last_audio_time']) > 0.5:  # 500ms silence
            session['silence_packets'] += 1
            if session['silence_packets'] >= self.end_threshold_packets:
                await self._end_recording(client_id)
        else:
            session['silence_packets'] = 0
    
    async def _end_recording(self, client_id: str, force: bool = False):
        """End current recording and process the audio"""
        if client_id not in self.recording_sessions or not self.recording_sessions[client_id]['is_recording']:
            return
        
        session = self.recording_sessions[client_id]
        session['is_recording'] = False
        
        if client_id in self.recording_timers:
            self.recording_timers[client_id].cancel()
            del self.recording_timers[client_id]
        
        if client_id in self.audio_buffers and self.audio_buffers[client_id]:
            audio_data = self.audio_buffers[client_id]
            self.audio_buffers[client_id] = b''
            metadata = self.buffer_metadata.get(client_id, {})
            player_name = metadata.get('player_name')
            session_id = metadata.get('session_id')
            profanity_words = metadata.get('profanity_words', [])
            
            await self._process_audio_async(client_id, audio_data, profanity_words, player_name, session_id)
            
            recording_duration = time.time() - session['start_time'] if session['start_time'] else 0
            logger.info(f"Ended recording for {client_id} (duration: {recording_duration:.1f}s, force: {force})")
    
    async def _max_duration_timeout(self, client_id: str):
        """Handle maximum recording duration timeout"""
        try:
            await asyncio.sleep(self.max_recording_duration_seconds)
            if client_id in self.recording_sessions and self.recording_sessions[client_id]['is_recording']:
                logger.info(f"Max recording duration reached for {client_id}, ending recording")
                await self._end_recording(client_id, force=True)
        except asyncio.CancelledError:
            pass
    
    def _estimate_audio_duration(self, audio_data: bytes) -> float:
        """Estimate audio duration in milliseconds from WAV data"""
        try:
            if len(audio_data) < 44:
                return 0.0
            
            import struct
            sample_rate = struct.unpack('<I', audio_data[24:28])[0]
            channels = struct.unpack('<H', audio_data[22:24])[0]
            audio_size = len(audio_data) - 44
            bytes_per_sample = 2  # 16-bit audio
            total_samples = audio_size // (channels * bytes_per_sample)
            duration_ms = (total_samples / sample_rate) * 1000
            
            return duration_ms
        except Exception as e:
            logger.debug(f"Failed to estimate audio duration: {e}")
            return 0.0
    
    def _should_process_audio(self, audio_data: bytes) -> bool:
        """Check if audio should be processed based on basic validation"""
        try:
            # Basic size check
            if len(audio_data) < 44:  # Minimum WAV header size
                return False
            
            # Estimate duration
            duration_ms = self._estimate_audio_duration(audio_data)
            min_duration = self.config.get("audio", {}).get("min_audio_length_ms", 50)
            
            # Allow processing if duration is at least the minimum
            return duration_ms >= min_duration
            
        except Exception as e:
            logger.warning(f"Audio validation check failed: {e}")
            return False
    
    
    async def _process_audio_async(self, client_id: str, audio_data: bytes, profanity_words: list, player_name: str = None, session_id: str = None):
        """Process audio asynchronously without blocking WebSocket"""
        try:
            # Check if client is still connected before processing
            if client_id not in self.active_connections:
                logger.debug(f"Client {client_id} disconnected before processing, skipping")
                return
                
            if self.transcriber:
                transcript = await asyncio.wait_for(
                    self.transcriber.transcribe(audio_data),
                    timeout=30.0
                )

                # Check again if client is still connected after transcription
                if client_id not in self.active_connections:
                    logger.debug(f"Client {client_id} disconnected during processing, skipping result")
                    return

                flagged_words = []
                if transcript:
                    combined_words = set(profanity_words or [])
                    if self.profanity_filter and hasattr(self.profanity_filter, 'words'):
                        try:
                            combined_words |= set(self.profanity_filter.words)
                        except Exception:
                            pass
                    temp_filter = ProfanityFilter(words=list(combined_words), case_sensitive=False, partial_match=True)
                    flagged_words = temp_filter.check_text(transcript)
                is_profane = len(flagged_words) > 0

                actual_player_name = player_name if player_name else client_id
                actual_session_id = session_id if session_id else client_id

                await self.send_message(client_id, {
                    "type": "final_transcript",
                    "transcript": transcript or "",
                    "flagged": is_profane,
                    "bad_words": flagged_words,
                    "player": actual_player_name,
                    "session_id": actual_session_id
                })
                
                logger.info(f"Audio processing completed for {actual_player_name}: '{transcript}' (flagged: {is_profane})")
                
            else:
                raise Exception("Transcriber not initialized")
                
        except asyncio.TimeoutError:
            logger.error(f"Audio processing timed out for {client_id}")
            await self.send_message(client_id, {
                "type": "error",
                "message": "Audio processing timed out",
                "status": "error"
            })
        except Exception as e:
            logger.error(f"Audio processing failed for {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "status": "error"
            })

ws_manager = SimpleWebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_state['config'] = config
        await ws_manager.initialize(config)
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        logger.info("Application shutting down")

config = load_config()

app = FastAPI(
    title="VoiceSentinel Processor",
    description="Real-time audio processing and transcription service",
    version="2.0.3",
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
    return {
        "status": "healthy",
        "service": "voicesentinel-processor",
        "version": "2.0.3"
    }

@app.get("/stats")
async def get_stats():
    return {
        "active_connections": len(ws_manager.active_connections),
        "transcriber_ready": ws_manager.transcriber is not None,
        "profanity_filter_ready": ws_manager.profanity_filter is not None
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type", "")
            
            if message_type == "auth":
                server_key = data.get("server_key", "")
                if server_key and len(server_key) >= 16:
                    await ws_manager.send_message(client_id, {
                        "type": "auth_success",
                        "message": "Authentication successful"
                    })
                    logger.info(f"Client {client_id} authenticated successfully")
                else:
                    await ws_manager.send_message(client_id, {
                        "type": "auth_failed",
                        "reason": "Invalid server key"
                    })
                    logger.warning(f"Client {client_id} authentication failed")
                    
            elif message_type == "audio_chunk":
                audio_b64 = data.get("audio_data", "")
                profanity_words_raw = data.get("profanity_words", [])
                player_name = data.get("player", client_id)
                session_id = data.get("session_id", client_id)
                is_final = data.get("is_final", False)
                
                profanity_words = []
                if isinstance(profanity_words_raw, str):
                    try:
                        import json
                        profanity_words = json.loads(profanity_words_raw)
                    except:
                        profanity_words = []
                elif isinstance(profanity_words_raw, list):
                    profanity_words = profanity_words_raw
                
                if audio_b64:
                    try:
                        import base64
                        audio_bytes = base64.b64decode(audio_b64)
                        
                        if is_final:
                            await ws_manager.process_audio_final(client_id, audio_bytes, profanity_words, player_name, session_id)
                        else:
                            await ws_manager.process_audio(client_id, audio_bytes, profanity_words, player_name, session_id)
                    except Exception as e:
                        logger.error(f"Failed to decode audio data: {e}")
                        await ws_manager.send_message(client_id, {
                            "type": "error",
                            "message": "Invalid audio data",
                            "status": "error"
                        })
                        
            elif message_type == "ping":
                await ws_manager.send_message(client_id, {
                    "type": "pong",
                    "timestamp": data.get("timestamp", 0)
                })
            
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