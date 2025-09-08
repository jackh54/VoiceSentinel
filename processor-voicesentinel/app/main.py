#!/usr/bin/env python3

import os
import sys
import json
import logging
import asyncio
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
            "port": 8000
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
    
    async def initialize(self, config):
        try:
            transcription_config = config.get("transcription", {})
            self.transcriber = WhisperTranscriber(
                model_name=transcription_config.get("model", "tiny.en"),
                language=transcription_config.get("language", "en"),
                config=config
            )
            self.profanity_filter = create_default_filter()
            logger.info("WebSocket manager initialized successfully")
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
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def process_audio(self, client_id: str, audio_data: bytes, profanity_words: list):
        try:
            await self.send_message(client_id, {
                "type": "status",
                "status": "processing"
            })
            if self.transcriber:
                transcript = await self.transcriber.transcribe(audio_data)
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

                await self.send_message(client_id, {
                    "type": "final_transcript",
                    "transcript": transcript or "",
                    "flagged": is_profane,
                    "bad_words": flagged_words,
                    "player": client_id,
                    "session_id": client_id
                })
                
                return transcript, is_profane
            else:
                raise Exception("Transcriber not initialized")
                
        except Exception as e:
            logger.error(f"Audio processing failed for {client_id}: {e}")
            await self.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "status": "error"
            })
            return None, False

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
    version="2.0.1",
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
        "version": "2.0.1"
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
                        await ws_manager.process_audio(client_id, audio_bytes, profanity_words)
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
        port=server_config.get("port", 8000),
        log_level="info",
        access_log=False
    )