#!/usr/bin/env python3
"""
VoiceSentinel Processor Backend
Real-time voice transcription and profanity filtering service

This service provides asynchronous audio processing with queue-based job management,
supporting multiple transcription engines and robust error handling.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
import multiprocessing
import glob
import tempfile
import argparse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import unquote
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_403_FORBIDDEN
from starlette.responses import Response

# Import our modules
from app.transcriber import WhisperTranscriber, VoskTranscriber
from app.decoder import OpusDecoder
from app.profanity import ProfanityFilter, create_default_filter

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/processor.log', mode='a') if os.path.exists('logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# System information
CPU_CORES = multiprocessing.cpu_count()
logger.info(f"System initialized with {CPU_CORES} CPU cores")

# Global state
rate_limit_store = defaultdict(list)
config = {}
transcriber = None
decoder = None
profanity_filter = None
processing_queue = None
results_storage = {}
processing_active = False
worker_stats = {}

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

def validate_config(config_data: Dict[str, Any]) -> None:
    """Validate configuration structure and values"""
    required_sections = ['server', 'security', 'transcription', 'processing', 'audio']
    
    for section in required_sections:
        if section not in config_data:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate server configuration
    server_config = config_data['server']
    if not isinstance(server_config.get('port'), int) or not (1 <= server_config.get('port') <= 65535):
        raise ConfigurationError("Server port must be an integer between 1 and 65535")
    
    # Validate processing configuration
    processing_config = config_data['processing']
    max_jobs = processing_config.get('max_concurrent_jobs', 4)
    if not isinstance(max_jobs, int) or max_jobs < 1 or max_jobs > 100:
        raise ConfigurationError("max_concurrent_jobs must be an integer between 1 and 100")
    
    # Validate audio configuration
    audio_config = config_data['audio']
    max_size = audio_config.get('max_total_size', 5242880)
    if not isinstance(max_size, int) or max_size < 1024 or max_size > 50 * 1024 * 1024:
        raise ConfigurationError("max_total_size must be between 1KB and 50MB")
    
    logger.info("Configuration validation passed")

def load_config() -> Dict[str, Any]:
    """Load and validate configuration from config.json"""
    config_path = Path(__file__).parent.parent / 'config.json'
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config_data = get_default_config()
        
        validate_config(config_data)
        
        # Log key configuration settings
        transcription_engine = config_data.get('transcription', {}).get('engine', 'unknown')
        max_workers = config_data.get('processing', {}).get('max_concurrent_jobs', 4)
        logger.info(f"Transcription engine: {transcription_engine}")
        logger.info(f"Max concurrent workers: {max_workers}")
        logger.info(f"Audio processing limits: {config_data.get('audio', {}).get('max_total_size', 0)} bytes")
        
        return config_data
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")

def get_default_config() -> Dict[str, Any]:
    """Get default configuration optimized for CPU processing"""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            }
        },
        "security": {
            "api_keys": [],
            "rate_limit": {
                "authenticated": {
                    "window_seconds": 60,
                    "max_requests": 1000,
                    "block_duration_seconds": 300
                },
                "unauthenticated": {
                    "window_seconds": 60,
                    "max_requests": 10,
                    "block_duration_seconds": 1800
                }
            }
        },
        "transcription": {
            "engine": "whisper",
            "model": "tiny.en",
            "language": "en",
            "timeout_seconds": 90,
            "cpu_threads": min(4, CPU_CORES),
            "options": {
                "fp16": False,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.7,
                "beam_size": 1,
                "best_of": 1,
                "suppress_blank": True,
                "initial_prompt": "Player voice chat audio containing speech."
            }
        },
        "processing": {
            "max_concurrent_jobs": min(4, CPU_CORES),
            "queue_warning_threshold": 200,
            "max_queue_size": 1000,
            "timeout_seconds": 90,
            "retry_attempts": 2,
            "retry_delay_seconds": 1
        },
        "audio": {
            "max_chunk_size": 262144,  # 256KB
            "max_total_size": 5242880,  # 5MB
            "sample_rate": 16000,
            "channels": 1,
            "min_length_ms": 300,
            "max_length_ms": 15000
        }
    }

# Initialize configuration
try:
    config = load_config()
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    exit(1)

def get_api_key(key: str = Query(None, description="API key for authentication")) -> str:
    """Validate API key and return authentication status"""
    if not config.get("security", {}).get("api_keys"):
        return key
        
    if key not in config["security"]["api_keys"]:
        return None  # Return None for unauthenticated requests
    return key

def check_rate_limit(api_key: str = Depends(get_api_key)):
    """Check rate limits with different thresholds for authenticated/unauthenticated requests"""
    if not config.get("server", {}).get("auth", {}).get("rate_limit"):
        return

    # Determine if request is authenticated
    is_authenticated = api_key in (config.get("security", {}).get("api_keys", []))
    
    # Get appropriate rate limit config
    rate_config = config["security"]["rate_limit"]
    if is_authenticated:
        limit_config = rate_config["authenticated"]
    else:
        limit_config = rate_config["unauthenticated"]
        if not api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
    
    window = limit_config["window_seconds"]
    max_requests = limit_config["max_requests"]
    block_duration = limit_config["block_duration_seconds"]
    
    now = datetime.now()
    rate_limit_store[api_key] = [
        ts for ts in rate_limit_store[api_key]
        if now - ts < timedelta(seconds=window)
    ]
    if rate_limit_store[api_key] and \
       len(rate_limit_store[api_key]) >= max_requests and \
       now - rate_limit_store[api_key][0] < timedelta(seconds=block_duration):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail=f"Rate limit exceeded. Try again in {block_duration} seconds."
        )
    
    rate_limit_store[api_key].append(now)
    return api_key

class TranscribeRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    player: str = Field(..., description="Player name")
    timestamp: int = Field(..., description="Unix timestamp")
    audio_format: str = Field(..., description="Audio format (wav)")
    audio_data: str = Field(..., description="Base64 encoded WAV file")
    chunk_count: Optional[int] = Field(None, description="Number of chunks combined")
    duration_ms: Optional[int] = Field(None, description="Duration of speech sequence in milliseconds")
    server_key: str = Field(..., description="Server key for file-based result delivery")
    profanity_words: List[str] = Field(default_factory=list, description="List of words to filter")
    case_sensitive: bool = Field(default=False, description="Whether profanity filtering is case sensitive")
    partial_match: bool = Field(default=True, description="Whether to match partial words")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v) < 8 or len(v) > 128:
            raise ValueError('Session ID must be between 8 and 128 characters')
        return v
    
    @validator('server_key')
    def validate_server_key(cls, v):
        if not v or len(v) < 16 or len(v) > 256:
            raise ValueError('Server key must be between 16 and 256 characters')
        return v
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if not v or len(v) > 10485760:  # 10MB limit
            raise ValueError('Audio data too large or empty')
        return v

class TranscribeResponse(BaseModel):
    session_id: str
    player: str
    flagged: bool
    bad_words: List[str]
    transcript: str
    chunks_processed: Optional[int] = None
    processing_time_ms: Optional[int] = None

class QueueResponse(BaseModel):
    """Response for queued transcription jobs"""
    session_id: str
    queued: bool
    queue_position: int
    estimated_wait_seconds: Optional[int] = None

class ProcessedResult(BaseModel):
    """Result stored for plugin polling"""
    session_id: str
    player: str
    timestamp: int
    flagged: bool
    bad_words: List[str]
    transcript: str
    processing_time_ms: int
    processed_at: str  # ISO timestamp
    error: Optional[str] = None

class CleanupRequest(BaseModel):
    """Request to remove processed results"""
    server_key: str = Field(..., description="Server key for authentication")
    session_ids: List[str] = Field(..., description="Session IDs to remove from results")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown with comprehensive resource management"""
    global processing_active, processing_queue
    
    logger.info("Starting VoiceSentinel Processor...")
    
    try:
        # Initialize core components
        await initialize_components()
        
        # Initialize processing queue
        processing_queue = asyncio.Queue()
        processing_active = True
        
        # Start background workers
        max_workers = config.get("processing", {}).get("max_concurrent_jobs", 4)
        processing_tasks = []
        
        for i in range(max_workers):
            task = asyncio.create_task(background_audio_processor(worker_id=i))
            processing_tasks.append(task)
            worker_stats[i] = {
                "jobs_processed": 0,
                "jobs_failed": 0,
                "total_processing_time": 0,
                "status": "idle",
                "last_activity": datetime.now().isoformat()
            }
        
        logger.info(f"Started {max_workers} parallel audio processing workers")
        logger.info("VoiceSentinel Processor is ready!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start VoiceSentinel Processor: {e}")
        raise
    
    finally:
        # Graceful shutdown
        logger.info("Shutting down VoiceSentinel Processor...")
        processing_active = False
        
        # Cancel all processing tasks
        for task in processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*processing_tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some processing tasks did not complete within timeout")
        except Exception as e:
            logger.warning(f"Error during task shutdown: {e}")
        
        # Cleanup components
        await cleanup_components()
        
        # Clean up temporary files
        cleanup_temporary_files()
        
        logger.info("VoiceSentinel Processor shutdown complete")

app = FastAPI(
    title="VoiceSentinel Processor",
    description="Real-time voice transcription and profanity filtering service",
    version="1.0.0",
    lifespan=lifespan
)

def setup_middleware(app: FastAPI):
    """Configure all middleware for the application"""
    # CORS middleware configuration
    cors_config = config.get("server", {}).get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses"""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    logger.info(f"CORS configured: origins={cors_config.get('allow_origins', ['*'])}")
    logger.info("Security middleware configured")

# Configure middleware
setup_middleware(app)

def cleanup_temporary_files():
    """Clean up temporary audio files from previous runs"""
    try:
        temp_dir = tempfile.gettempdir()
        
        # Find and remove any audio temp files
        patterns = [
            os.path.join(temp_dir, "tmp*voicesentinel*.wav"),
            os.path.join(temp_dir, "tmp*voicesentinel*.opus"),
            os.path.join(temp_dir, "tmp*voicesentinel*.ogg"),
        ]
        
        removed_count = 0
        for pattern in patterns:
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        # Check if file is older than 1 hour (safe cleanup)
                        import time
                        if time.time() - os.path.getmtime(file_path) > 3600:
                            os.unlink(file_path)
                            removed_count += 1
                except Exception as e:
                    logger.debug(f"Could not remove temp file {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stray temporary audio files")
        
    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")

async def initialize_components():
    """Initialize transcriber, decoder, and profanity filter with error handling"""
    global transcriber, decoder, profanity_filter
    
    try:
        # Clean up any existing temporary files
        cleanup_temporary_files()
        
        # Initialize transcriber
        transcriber_config = config.get("transcription", {})
        transcriber_type = transcriber_config.get("engine", "whisper")
        
        logger.info(f"Initializing {transcriber_type} transcriber...")
        
        if transcriber_type == "whisper":
            transcriber = WhisperTranscriber(
                model_name=transcriber_config.get("model", "tiny.en"),
                language=transcriber_config.get("language", "en"),
                config=config
            )
        elif transcriber_type == "vosk":
            transcriber = VoskTranscriber(
                model_path=transcriber_config.get("model_path", "vosk-model")
            )
        else:
            raise ValueError(f"Unsupported transcriber type: {transcriber_type}")
        
        # Initialize decoder
        logger.info("Initializing audio decoder...")
        decoder = OpusDecoder()
        
        # Initialize profanity filter
        logger.info("Initializing profanity filter...")
        profanity_filter = create_default_filter()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

async def cleanup_components():
    """Clean up all components properly"""
    global transcriber, decoder, profanity_filter
    
    try:
        if transcriber:
            await transcriber.cleanup()
            transcriber = None
        
        if decoder:
            await decoder.cleanup()
            decoder = None
        
        if profanity_filter:
            profanity_filter = None
        
        logger.info("All components cleaned up successfully")
        
    except Exception as e:
        logger.warning(f"Error during component cleanup: {e}")

async def background_audio_processor(worker_id: int = 0):
    """
    Background worker for processing audio transcription jobs
    
    Features:
    - Robust error handling with retry logic
    - Performance monitoring and statistics
    - Graceful shutdown handling
    - Resource cleanup
    """
    logger.info(f"Worker {worker_id} started")
    
    while processing_active:
        try:
            # Get job from queue with timeout
            job = await asyncio.wait_for(processing_queue.get(), timeout=1.0)
            
            # Update worker status
            worker_stats[worker_id]["status"] = "processing"
            worker_stats[worker_id]["last_activity"] = datetime.now().isoformat()
            
            session_id = job.get('session_id', 'unknown')
            player = job.get('player', 'unknown')
            
            logger.info(f"Worker {worker_id} processing job: {session_id} from {player}")
            
            # Process the job with comprehensive error handling
            try:
                await process_audio_job_with_retry(job, worker_id)
                worker_stats[worker_id]["jobs_processed"] += 1
                logger.debug(f"Worker {worker_id} completed job {session_id}")
                
            except Exception as e:
                worker_stats[worker_id]["jobs_failed"] += 1
                logger.error(f"Worker {worker_id} failed to process job {session_id}: {e}")
                
                # Store error result for the client
                error_result = ProcessedResult(
                    session_id=session_id,
                    player=player,
                    timestamp=job.get('timestamp', int(datetime.now().timestamp())),
                    flagged=False,
                    bad_words=[],
                    transcript="",
                    processing_time_ms=0,
                    processed_at=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z',
                    error=f"Processing failed: {str(e)}"
                )
                
                try:
                    await store_result(job.get('server_key', ''), error_result)
                except Exception as store_error:
                    logger.error(f"Failed to store error result: {store_error}")
            
            finally:
                # Mark job as done and update worker status
                processing_queue.task_done()
                worker_stats[worker_id]["status"] = "idle"
                worker_stats[worker_id]["last_activity"] = datetime.now().isoformat()
            
        except asyncio.TimeoutError:
            # No job available, continue polling
            continue
            
        except asyncio.CancelledError:
            # Worker is being shut down
            logger.info(f"Worker {worker_id} received shutdown signal")
            break
            
        except Exception as e:
            logger.error(f"Worker {worker_id} unexpected error: {e}")
            worker_stats[worker_id]["status"] = "error"
            worker_stats[worker_id]["last_activity"] = datetime.now().isoformat()
            
            # Brief pause before retrying to avoid rapid error loops
            await asyncio.sleep(1)
    
    logger.info(f"Worker {worker_id} stopped")

async def process_audio_job_with_retry(job: Dict[str, Any], worker_id: int):
    """Process audio job with retry logic and timeout handling"""
    max_retries = config.get("processing", {}).get("retry_attempts", 2)
    retry_delay = config.get("processing", {}).get("retry_delay_seconds", 1)
    
    for attempt in range(max_retries + 1):
        try:
            await process_audio_job(job, worker_id)
            return  # Success, exit retry loop
            
        except asyncio.TimeoutError:
            logger.error(f"Worker {worker_id} job timeout on attempt {attempt + 1}")
            if attempt == max_retries:
                raise  # Final attempt failed
            await asyncio.sleep(retry_delay * (attempt + 1))
            
        except Exception as e:
            logger.error(f"Worker {worker_id} job failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise  # Final attempt failed
            await asyncio.sleep(retry_delay * (attempt + 1))

async def process_audio_job(job: Dict[str, Any], worker_id: int):
    """Process a single audio transcription job with comprehensive error handling"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Add timeout for individual job processing
        timeout = config.get("processing", {}).get("timeout_seconds", 90)
        
        result = await asyncio.wait_for(
            _process_audio_job_internal(job, worker_id),
            timeout=timeout
        )
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        worker_stats[worker_id]["total_processing_time"] += processing_time
        
        logger.info(f"Worker {worker_id} completed job {job['session_id']} in {processing_time}ms")
        
    except asyncio.TimeoutError:
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        logger.error(f"Worker {worker_id} job timeout after {timeout}s for session {job['session_id']}")
        
        # Store timeout result
        timeout_result = ProcessedResult(
            session_id=job['session_id'],
            player=job['player'],
            timestamp=job['timestamp'],
            flagged=False,
            bad_words=[],
            transcript="",
            processing_time_ms=processing_time,
            processed_at=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z',
            error=f"Processing timeout after {timeout} seconds"
        )
        await store_result(job['server_key'], timeout_result)
        raise

async def _process_audio_job_internal(job: Dict[str, Any], worker_id: int):
    """Internal job processing logic with enhanced error handling and validation"""
    start_time = asyncio.get_event_loop().time()
    
    # Extract job parameters with validation
    session_id = job.get('session_id')
    player = job.get('player')
    timestamp = job.get('timestamp')
    audio_data = job.get('audio_data')
    profanity_words = job.get('profanity_words', [])
    case_sensitive = job.get('case_sensitive', False)
    partial_match = job.get('partial_match', True)
    server_key = job.get('server_key')
    
    # Validate required parameters
    if not all([session_id, player, timestamp, audio_data, server_key]):
        raise ValueError("Missing required job parameters")
    
    # Decode base64 audio with validation
    try:
        decoded_audio = base64.b64decode(audio_data)
        logger.debug(f"Decoded WAV file: {len(decoded_audio)} bytes")
        
        # Validate audio data
        if len(decoded_audio) < 44:
            raise ValueError("Audio data too small (missing WAV header)")
        
        # Check for valid WAV header
        if not (decoded_audio[0:4] == b'RIFF' and decoded_audio[8:12] == b'WAVE'):
            raise ValueError("Invalid WAV file format")
            
    except Exception as e:
        logger.error(f"Failed to decode/validate audio data: {e}")
        raise ValueError(f"Invalid audio data: {e}")
    
    # Transcribe audio with error handling
    try:
        transcript = await transcriber.transcribe(decoded_audio)
        logger.debug(f"Transcription result: '{transcript}'")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise ValueError(f"Transcription error: {e}")
    
    # Check for profanity
    flagged_words = []
    if transcript and profanity_words:
        try:
            # Create temporary profanity filter for this request
            temp_filter = ProfanityFilter(
                words=profanity_words,
                case_sensitive=case_sensitive,
                partial_match=partial_match
            )
            flagged_words = temp_filter.check_text(transcript)
            
            logger.debug(f"Profanity check: {len(flagged_words)} words flagged")
            
        except Exception as e:
            logger.error(f"Profanity filtering failed: {e}")
            # Continue processing without profanity check
    
    processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
    
    # Log results
    if flagged_words:
        logger.warning(f"FLAGGED CONTENT from {player}: {flagged_words}")
    
    logger.info(f"Transcribed '{transcript}' for {player} in {processing_time}ms")
    
    # Create and store result
    result = ProcessedResult(
        session_id=session_id,
        player=player,
        timestamp=timestamp,
        flagged=bool(flagged_words),
        bad_words=flagged_words,
        transcript=transcript,
        processing_time_ms=processing_time,
        processed_at=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z'
    )
    
    await store_result(server_key, result)
    return result

async def store_result(server_key, result: ProcessedResult):
    """Store transcription result in memory for web API polling"""
    try:
        # Initialize server results if not exists
        if server_key not in results_storage:
            results_storage[server_key] = {}
        
        # Add result to in-memory storage
        results_storage[server_key][result.session_id] = result
        
        logger.info(f"Stored result for {result.session_id} in memory - server {server_key} now has {len(results_storage[server_key])} pending results")
        
    except Exception as e:
        logger.error(f"Failed to store result: {e}")

@app.get("/", dependencies=[Depends(check_rate_limit)])
async def root(key: str = Depends(get_api_key)):
    return {"message": "VoiceSentinel Processor is running"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "service": "VoiceSentinel Processor",
        "status": "healthy",
        "components": {
            "transcriber": transcriber is not None,
            "decoder": decoder is not None,
            "profanity_filter": profanity_filter is not None
        },
        "config": {
            "transcriber_type": config.get("transcription", {}).get("engine", "unknown"),
            "profanity_words_count": len(config.get("profanity", {}).get("words", []))
        },
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/transcribe", response_model=QueueResponse, dependencies=[Depends(check_rate_limit)])
async def queue_audio_transcription(request: TranscribeRequest, key: str = Depends(get_api_key)):
    """
    Queue WAV audio for async transcription processing
    Returns immediately - results available via file polling
    """
    try:
        if config.get("security", {}).get("api_keys") and key not in config["security"]["api_keys"]:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")
        
        extra_info = ""
        if request.chunk_count is not None:
            extra_info += f", combined from {request.chunk_count} chunks"
        if request.duration_ms is not None:
            extra_info += f", duration: {request.duration_ms}ms"
        
        logger.info(f"Queueing WAV audio from player: {request.player} (session: {request.session_id}){extra_info}")
        logger.info(f"Profanity words received: {request.profanity_words} (count: {len(request.profanity_words)})")
        logger.info(f"Case sensitive: {request.case_sensitive}, Partial match: {request.partial_match}")
        if request.audio_format.lower() != "wav":
            raise HTTPException(status_code=400, detail="Only WAV audio format is supported")
        
        if not request.server_key or len(request.server_key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key")
        try:
            wav_data = base64.b64decode(request.audio_data)
            logger.info(f"Decoded WAV file: {len(wav_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to decode base64 WAV data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 WAV data")
        
        if len(wav_data) < 44:
            raise HTTPException(status_code=400, detail="WAV file too small (missing header)")
        
        if not (wav_data[0:4] == b'RIFF' and wav_data[8:12] == b'WAVE'):
            raise HTTPException(status_code=400, detail="Invalid WAV file header")
        job = {
            'session_id': request.session_id,
            'player': request.player,
            'timestamp': request.timestamp,
            'server_key': request.server_key,
            'audio_data': request.audio_data,
            'profanity_words': request.profanity_words,
            'case_sensitive': request.case_sensitive,
            'partial_match': request.partial_match,
            'chunk_count': request.chunk_count,
            'duration_ms': request.duration_ms
        }
        
        current_queue_size = processing_queue.qsize()
        max_queue_size = config.get("processing", {}).get("max_queue_size", 2000)
        
        if current_queue_size >= max_queue_size:
            logger.warning(f"Queue full ({current_queue_size}/{max_queue_size}), rejecting new job from {request.player}")
            raise HTTPException(
                status_code=503, 
                detail=f"Queue is full ({current_queue_size}/{max_queue_size}). Please try again later."
            )
        
        await processing_queue.put(job)
        queue_size = processing_queue.qsize()
        max_workers = config.get("processing", {}).get("max_concurrent_jobs", 10)
        estimated_wait = max(1, queue_size // max_workers) * 2
        
        logger.info(f"Queued audio job {request.session_id} for {request.player} (queue size: {queue_size})")
        
        return QueueResponse(
            session_id=request.session_id,
            queued=True,
            queue_position=queue_size,
            estimated_wait_seconds=estimated_wait
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error queueing audio: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/results")
async def get_pending_results(key: str = Query(..., description="Server key for authentication")):
    """
    Get pending transcription results for a server key
    """
    try:
        if not key or len(key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key")
        
        decoded_key = unquote(key)
        
        logger.info(f"Looking up results for key: '{decoded_key}' (original: '{key}', length: {len(decoded_key)})")
        logger.info(f"Available server keys in storage: {list(results_storage.keys())}")
        
        if decoded_key not in results_storage:
            logger.warning(f"No results found for server key: '{decoded_key}'")
            return {"results": {}, "count": 0}
        
        server_results = results_storage[decoded_key]
        results_data = {
            session_id: result.dict() 
            for session_id, result in server_results.items()
        }
        
        logger.info(f"Serving {len(results_data)} pending results for server {decoded_key}")
        
        return {
            "results": results_data,
            "count": len(results_data),
            "server_key": decoded_key,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get results")

@app.get("/removeresult")
@app.post("/removeresult") 
@app.delete("/removeresult")
async def remove_processed_results_all_methods(
    request: Request,
    server_key: str = Query(None, description="Server key for authentication"),
    key: str = Query(None, description="Alternative server key parameter"),
    session_ids: str = Query(None, description="Comma-separated session IDs to remove")
):
    """
    Remove processed results from storage - supports GET, POST, and DELETE methods
    """
    try:
        method = request.method
        logger.info(f"{method} cleanup request received - key param: {key}, server_key param: {server_key}, session_ids: {session_ids}")
        
        # Use either server_key or key parameter
        auth_key = server_key or key
        
        if not auth_key or len(auth_key) < 8:
            logger.error(f"Invalid server key - auth_key: {auth_key}")
            raise HTTPException(status_code=400, detail="Invalid server key - provide either 'key' or 'server_key' parameter")
        
        decoded_key = unquote(auth_key)
        
        logger.info(f"{method} cleanup request - decoded key: {decoded_key}, session_ids: {session_ids}")
        
        if decoded_key not in results_storage:
            logger.warning(f"No results found for server key: {decoded_key}")
            return {"removed": 0, "message": "No results found for server key", "remaining": 0}
        
        server_results = results_storage[decoded_key]
        if not session_ids:
            removed_count = len(server_results)
            del results_storage[decoded_key]
            logger.info(f"Removed ALL {removed_count} processed results for server {decoded_key}")
            return {
                "removed": removed_count,
                "remaining": 0,
                "message": f"Successfully removed all {removed_count} processed results"
            }
        
        session_id_list = [sid.strip() for sid in session_ids.split(",") if sid.strip()]
        
        removed_count = 0
        for session_id in session_id_list:
            if session_id in server_results:
                del server_results[session_id]
                removed_count += 1
        
        remaining_count = len(server_results)
        if remaining_count == 0:
            del results_storage[decoded_key]
        
        logger.info(f"Removed {removed_count} processed results for server, {remaining_count} remaining")
        
        return {
            "removed": removed_count,
            "remaining": remaining_count,
            "message": f"Successfully removed {removed_count} processed results"
        }
        
    except Exception as e:
        logger.error(f"Error removing results: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove results")

@app.get("/stats", dependencies=[Depends(check_rate_limit)])
async def get_stats(key: str = Depends(get_api_key)):
    """Get processing statistics"""
    queue_size = processing_queue.qsize()
    total_results = sum(len(results) for results in results_storage.values())
    
    active_workers = sum(1 for stats in worker_stats.values() if stats["status"] == "processing")
    total_processed = sum(stats["jobs_processed"] for stats in worker_stats.values())
    total_failed = sum(stats["jobs_failed"] for stats in worker_stats.values())
    
    return {
        "status": "running",
        "queue_size": queue_size,
        "queue_warning_threshold": config.get("processing", {}).get("queue_warning_threshold", 300),
        "max_queue_size": config.get("processing", {}).get("max_queue_size", 5000),
        "total_results_pending": total_results,
        "workers": {
            "total": len(worker_stats),
            "active": active_workers,
            "idle": len(worker_stats) - active_workers,
            "jobs_processed": total_processed,
            "jobs_failed": total_failed,
            "success_rate": f"{(total_processed / max(1, total_processed + total_failed)) * 100:.1f}%"
        },
        "processing": {
            "timeout_seconds": config.get("processing", {}).get("timeout_seconds", 90),
            "retry_attempts": config.get("processing", {}).get("retry_attempts", 2),
            "max_concurrent_jobs": config.get("processing", {}).get("max_concurrent_jobs", 25)
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=None, help='Host to bind to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=None, help='Number of HTTP worker processes')
    args = parser.parse_args()
    
    load_config()
    
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = args.port or server_config.get("port", 8000)
    workers = args.workers or server_config.get("workers", 1)
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    ) 