#!/usr/bin/env python3
"""
VoiceSentinel Processor Backend
Real-time voice transcription and profanity filtering service
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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import unquote
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.status import HTTP_403_FORBIDDEN

# Import our modules
from app.transcriber import WhisperTranscriber, VoskTranscriber
from app.decoder import OpusDecoder
from app.profanity import ProfanityFilter, create_default_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Detect CPU cores and optimize for CPU processing
CPU_CORES = multiprocessing.cpu_count()
logger.info(f"Detected {CPU_CORES} CPU cores on this system")

# Authentication and Rate Limiting
api_key_header = APIKeyHeader(name="X-API-Key")
rate_limit_store = defaultdict(list)  # Store request timestamps per API key

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate API key and return authentication status"""
    if not config.get("server", {}).get("auth", {}).get("api_keys"):
        return api_key_header
        
    if api_key_header not in config["server"]["auth"]["api_keys"]:
        return None  # Return None for unauthenticated requests
    return api_key_header

def check_rate_limit(api_key: str = Depends(get_api_key)):
    """Check rate limits with different thresholds for authenticated/unauthenticated requests"""
    if not config.get("server", {}).get("auth", {}).get("rate_limit"):
        return

    # Determine if request is authenticated
    is_authenticated = api_key in (config.get("server", {}).get("auth", {}).get("api_keys", []))
    
    # Get appropriate rate limit config
    rate_config = config["server"]["auth"]["rate_limit"]
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
    
    # Clean old requests
    now = datetime.now()
    rate_limit_store[api_key] = [
        ts for ts in rate_limit_store[api_key]
        if now - ts < timedelta(seconds=window)
    ]
    
    # Check if blocked
    if rate_limit_store[api_key] and \
       len(rate_limit_store[api_key]) >= max_requests and \
       now - rate_limit_store[api_key][0] < timedelta(seconds=block_duration):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail=f"Rate limit exceeded. Try again in {block_duration} seconds."
        )
    
    # Add current request
    rate_limit_store[api_key].append(now)
    return api_key

# Request/Response models

class TranscribeRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    player: str = Field(..., description="Player name")
    timestamp: int = Field(..., description="Unix timestamp")
    audio_format: str = Field(..., description="Audio format (wav)")
    audio_data: str = Field(..., description="Base64 encoded WAV file")
    chunk_count: Optional[int] = Field(None, description="Number of chunks combined")
    duration_ms: Optional[int] = Field(None, description="Duration of speech sequence in milliseconds")
    
    # Server identification and async processing
    server_key: str = Field(..., description="Server key for file-based result delivery")
    
    # Profanity filtering configuration
    profanity_words: List[str] = Field(default_factory=list, description="List of words to filter")
    case_sensitive: bool = Field(default=False, description="Whether profanity filtering is case sensitive")
    partial_match: bool = Field(default=True, description="Whether to match partial words")

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

# Initialize global components
config = {}
transcriber = None
decoder = None
profanity_filter = None

# Async processing queue and results storage
processing_queue = asyncio.Queue()
results_storage = {}  # server_key -> {session_id -> ProcessedResult}
processing_active = False
worker_stats = {}  # worker_id -> stats

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global processing_active
    
    # Startup
    logger.info("Starting VoiceSentinel Processor...")
    load_config()
    initialize_components()
    
    # Start multiple parallel processing workers
    processing_active = True
    max_workers = config.get("processing", {}).get("max_concurrent_jobs", 25)
    processing_tasks = []
    
    for i in range(max_workers):
        task = asyncio.create_task(background_audio_processor(worker_id=i))
        processing_tasks.append(task)
        worker_stats[i] = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0,
            "status": "idle"
        }
    
    logger.info(f"Started {max_workers} parallel audio processing workers")
    
    # Results now stored in memory and served via web API
    logger.info("Using in-memory storage for transcription results")
    
    logger.info("VoiceSentinel Processor is ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VoiceSentinel Processor...")
    processing_active = False
    
    # Cancel all background tasks
    for task in processing_tasks:
        task.cancel()
    
    # Wait for all tasks to complete
    try:
        await asyncio.gather(*processing_tasks, return_exceptions=True)
    except Exception as e:
        logger.warning(f"Error during task shutdown: {e}")
    
    if transcriber:
        await transcriber.cleanup()
    if decoder:
        await decoder.cleanup()
    
    logger.info("VoiceSentinel Processor shutdown complete")

# Global components
app = FastAPI(
    title="VoiceSentinel Processor",
    description="Real-time voice transcription and profanity filtering service",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(check_rate_limit)]
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config():
    """Load configuration from config.json"""
    global config
    config_path = os.path.join(os.path.dirname(__file__), '../config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration optimized for CPU processing and high concurrency (200+ players)
        # Use CPU cores efficiently: leave 1-2 cores for system, use rest for workers
        optimal_workers = max(8, min(CPU_CORES * 2, 32))  # Increased from 16 to 32 max workers
        
        config = {
            "transcriber": {
                "type": "whisper",
                "model_name": "tiny.en",
                "language": "en"
            },
            "whisper": {
                "timeout_seconds": 90,
                "fp16": False,  # CPU processing
                "threads": 0,   # Let Whisper auto-detect optimal CPU threads
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.7,
                "beam_size": 1,
                "best_of": 1
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "processing": {
                "max_concurrent_jobs": optimal_workers,
                "queue_warning_threshold": optimal_workers * 20,  # Increased from 10 to 20
                "max_queue_size": optimal_workers * 100,  # Increased from 50 to 100
                "processing_timeout_seconds": 90,
                "retry_attempts": 1,
                "retry_delay_seconds": 1
            },
            "audio": {
                "max_chunk_size": 262144,
                "max_total_size": 5242880,
                "sample_rate": 16000,
                "channels": 1,
                "min_audio_length_ms": 300,
                "max_audio_length_ms": 15000
            }
        }
    
    # Override worker count with CPU-optimized value if not explicitly set
    if 'processing' in config:
        current_workers = config['processing'].get('max_concurrent_jobs', 0)
        if current_workers > CPU_CORES * 2:
            # Cap at 2x CPU cores to prevent oversubscription
            optimal_workers = CPU_CORES * 2
            config['processing']['max_concurrent_jobs'] = optimal_workers
            logger.info(f"Adjusted worker count from {current_workers} to {optimal_workers} (CPU optimized)")
    
    logger.info(f"CPU Cores: {CPU_CORES}, Workers: {config['processing']['max_concurrent_jobs']}")
    logger.info(f"Configuration loaded and optimized for CPU processing")

def cleanup_temporary_files():
    """Clean up any temporary audio files that might be left from previous runs"""
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
            logger.info(f"PRIVACY: Cleaned up {removed_count} stray temporary audio files")
        
    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")

def initialize_components():
    """Initialize transcriber, decoder, and profanity filter"""
    global transcriber, decoder, profanity_filter
    
    # Clean up any leftover temporary files first
    cleanup_temporary_files()
    
    # Initialize transcriber
    transcriber_config = config.get("transcriber", {})
    transcriber_type = transcriber_config.get("type", "whisper")
    
    if transcriber_type == "whisper":
        transcriber = WhisperTranscriber(
            model_name=transcriber_config.get("model_name", "tiny.en"),
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
    decoder = OpusDecoder()
    
    # Initialize profanity filter
    profanity_filter = create_default_filter()
    
    logger.info("Components initialized successfully")

async def background_audio_processor(worker_id: int = 0):
    """Background worker for processing audio transcription jobs"""
    logger.info(f"Worker {worker_id} started")
    
    while processing_active:
        try:
            # Get job from queue with timeout
            job = await asyncio.wait_for(processing_queue.get(), timeout=1.0)
            
            worker_stats[worker_id]["status"] = "processing"
            logger.info(f"Worker {worker_id} processing job: {job['session_id']} from {job['player']}")
            
            # Process the job with retry logic
            await process_audio_job_with_retry(job, worker_id)
            
            # Mark job as done
            processing_queue.task_done()
            worker_stats[worker_id]["status"] = "idle"
            
        except asyncio.TimeoutError:
            # No job available, continue polling
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            worker_stats[worker_id]["status"] = "error"
            await asyncio.sleep(1)  # Brief pause before retrying
    
    logger.info(f"Worker {worker_id} stopped")

async def process_audio_job_with_retry(job, worker_id: int):
    """Process audio job with retry logic"""
    max_retries = config.get("processing", {}).get("retry_attempts", 2)
    retry_delay = config.get("processing", {}).get("retry_delay_seconds", 1)
    
    for attempt in range(max_retries + 1):
        try:
            await process_audio_job(job, worker_id)
            worker_stats[worker_id]["jobs_processed"] += 1
            return
            
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Worker {worker_id} job retry {attempt + 1}/{max_retries} for {job['session_id']}: {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Worker {worker_id} job failed after {max_retries} retries for {job['session_id']}: {e}")
                worker_stats[worker_id]["jobs_failed"] += 1
                
                # Store error result
                error_result = ProcessedResult(
                    session_id=job['session_id'],
                    player=job['player'],
                    timestamp=job['timestamp'],
                    flagged=False,
                    bad_words=[],
                    transcript="",
                    processing_time_ms=0,
                    processed_at=datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z',
                    error=f"Processing failed after {max_retries} retries: {str(e)}"
                )
                await store_result(job['server_key'], error_result)

async def process_audio_job(job, worker_id: int):
    """Process a single audio transcription job"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Add timeout for individual job processing
        timeout = config.get("processing", {}).get("processing_timeout_seconds", 180)
        
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

async def _process_audio_job_internal(job, worker_id: int):
    """Internal job processing logic"""
    start_time = asyncio.get_event_loop().time()
    
    session_id = job['session_id']
    player = job['player']
    timestamp = job['timestamp']
    audio_data = job['audio_data']
    profanity_words = job['profanity_words']
    case_sensitive = job['case_sensitive']
    partial_match = job['partial_match']
    server_key = job['server_key']
    
    # Decode base64 audio
    try:
        decoded_audio = base64.b64decode(audio_data)
        logger.info(f"Decoded WAV file: {len(decoded_audio)} bytes")
    except Exception as e:
        logger.error(f"Failed to decode audio data: {e}")
        raise
    
    # Transcribe audio
    transcript = await transcriber.transcribe(decoded_audio)
    
    # Check for profanity
    flagged_words = []
    if transcript and profanity_words:
        # Create temporary profanity filter for this request
        temp_filter = ProfanityFilter(
            words=profanity_words,
            case_sensitive=case_sensitive,
            partial_match=partial_match
        )
        flagged_words = temp_filter.check_text(transcript)
    
    processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
    
    # Log results
    if flagged_words:
        logger.warning(f"FLAGGED CONTENT from {player}: {flagged_words}")
    
    logger.info(f"Transcribed '{transcript}' for {player} in {processing_time}ms")
    
    # Store result
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

@app.get("/")
async def root(api_key: str = Depends(get_api_key)):
    return {"message": "VoiceSentinel Processor is running"}

@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
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
            "transcriber_type": config.get("transcriber", {}).get("type", "unknown"),
            "profanity_words_count": len(config.get("profanity", {}).get("words", []))
        },
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/transcribe", response_model=QueueResponse)
async def queue_audio_transcription(request: TranscribeRequest, api_key: str = Depends(get_api_key)):
    """
    Queue WAV audio for async transcription processing
    Returns immediately - results available via file polling
    """
    try:
        extra_info = ""
        if request.chunk_count is not None:
            extra_info += f", combined from {request.chunk_count} chunks"
        if request.duration_ms is not None:
            extra_info += f", duration: {request.duration_ms}ms"
        
        logger.info(f"Queueing WAV audio from player: {request.player} (session: {request.session_id}){extra_info}")
        
        # Validate audio format
        if request.audio_format.lower() != "wav":
            raise HTTPException(status_code=400, detail="Only WAV audio format is supported")
        
        # Validate server key
        if not request.server_key or len(request.server_key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key")
        
        # Decode base64 WAV data
        try:
            wav_data = base64.b64decode(request.audio_data)
            logger.info(f"Decoded WAV file: {len(wav_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to decode base64 WAV data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 WAV data")
        
        # Validate WAV file header
        if len(wav_data) < 44:
            raise HTTPException(status_code=400, detail="WAV file too small (missing header)")
        
        if not (wav_data[0:4] == b'RIFF' and wav_data[8:12] == b'WAVE'):
            raise HTTPException(status_code=400, detail="Invalid WAV file header")
        
        # WAV file saving permanently disabled for privacy/security
        # Audio data is processed in memory only - no persistent storage
        
        # Create job for background processing
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
        
        # Check queue size and reject if too full
        current_queue_size = processing_queue.qsize()
        max_queue_size = config.get("processing", {}).get("max_queue_size", 2000)
        
        if current_queue_size >= max_queue_size:
            logger.warning(f"Queue full ({current_queue_size}/{max_queue_size}), rejecting new job from {request.player}")
            raise HTTPException(
                status_code=503, 
                detail=f"Queue is full ({current_queue_size}/{max_queue_size}). Please try again later."
            )
        
        # Add to processing queue
        await processing_queue.put(job)
        queue_size = processing_queue.qsize()
        
        # Estimate wait time based on queue position and concurrent workers
        max_workers = config.get("processing", {}).get("max_concurrent_jobs", 10)
        estimated_wait = max(1, queue_size // max_workers) * 2  # Rough estimate
        
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
async def get_pending_results(key: str, api_key: str = Depends(get_api_key)):
    """
    Get pending transcription results for a server key
    """
    try:
        if not key or len(key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key")
        
        # URL decode the server key since it gets encoded in HTTP requests
        decoded_key = unquote(key)
        
        # Debug logging to understand the key lookup issue
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

@app.post("/removeresult")
async def remove_processed_results(key: str, request: CleanupRequest, api_key: str = Depends(get_api_key)):
    """
    Remove processed results from storage after plugin has handled them
    """
    try:
        if not key or len(key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key in URL")
        
        # URL decode the server key since it gets encoded in HTTP requests
        decoded_key = unquote(key)
        
        if not request.server_key or len(request.server_key) < 8:
            raise HTTPException(status_code=400, detail="Invalid server key in body")
        
        # Verify URL key matches body key for security
        if decoded_key != request.server_key:
            raise HTTPException(status_code=400, detail="Server key mismatch")
        
        if decoded_key not in results_storage:
            return {"removed": 0, "message": "No results found for server key", "remaining": 0}
        
        removed_count = 0
        server_results = results_storage[decoded_key]
        
        # Remove requested session IDs
        for session_id in request.session_ids:
            if session_id in server_results:
                del server_results[session_id]
                removed_count += 1
        
        # Clean up empty server storage
        remaining_count = len(server_results)
        if remaining_count == 0:
            del results_storage[decoded_key]
        
        logger.info(f"Removed {removed_count} processed results for server {decoded_key}, {remaining_count} remaining")
        
        return {
            "removed": removed_count,
            "remaining": remaining_count,
            "message": f"Successfully removed {removed_count} processed results"
        }
        
    except Exception as e:
        logger.error(f"Error removing results: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove results")

@app.get("/stats")
async def get_stats(api_key: str = Depends(get_api_key)):
    """Get processing statistics"""
    queue_size = processing_queue.qsize()
    total_results = sum(len(results) for results in results_storage.values())
    
    # Calculate worker statistics
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
            "timeout_seconds": config.get("processing", {}).get("processing_timeout_seconds", 180),
            "retry_attempts": config.get("processing", {}).get("retry_attempts", 2),
            "max_concurrent_jobs": config.get("processing", {}).get("max_concurrent_jobs", 25)
        }
    }

if __name__ == "__main__":
    # Load configuration
    load_config()
    
    # Get server config
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    workers = server_config.get("workers", 1)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    ) 