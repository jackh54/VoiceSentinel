"""
Audio transcription module using OpenAI Whisper or Vosk
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import json
import wave
import struct
from abc import ABC, abstractmethod
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BaseTranscriber(ABC):
    """Base class for audio transcribers"""
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass

class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper based transcriber with enhanced error handling"""
    
    def __init__(self, model_name: str = "small.en", language: str = "en", config: dict = None):
        self.model_name = model_name
        self.language = language
        self.config = config or {}
        self.model = None
        self._load_model()
        
        # Whisper-specific configuration
        self.whisper_config = self.config.get("whisper", {})
        self.timeout_seconds = self.whisper_config.get("timeout_seconds", 150)
        
        logger.info(f"Initialized OpenAI Whisper transcriber with model: {model_name}")
    
    def _load_model(self):
        """Load the Whisper model with CPU optimization"""
        try:
            import whisper
            import torch
            
            # Set CPU processing for optimal performance
            threads = self.config.get("whisper", {}).get("threads", 0)
            if threads > 0:
                torch.set_num_threads(threads)
            
            # Force CPU processing (no CUDA)
            device = "cpu"
            
            self.model = whisper.load_model(self.model_name, device=device)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully on CPU")
            
            # Log CPU optimization info
            num_threads = torch.get_num_threads()
            logger.info(f"PyTorch using {num_threads} CPU threads for Whisper processing")
            
        except ImportError:
            logger.error("OpenAI Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _validate_audio(self, audio_data: bytes) -> Tuple[bool, str]:
        """Validate audio data before processing"""
        try:
            if len(audio_data) < 44:  # Minimum WAV header size
                return False, "Audio data too short (less than WAV header size)"
            
            # Check if it's a valid WAV file
            if not audio_data.startswith(b'RIFF') or b'WAVE' not in audio_data[:12]:
                return False, "Invalid WAV file format"
            
            # Parse WAV header to get basic info
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                try:
                    with wave.open(temp_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        
                        # Check for reasonable parameters
                        if frames == 0:
                            return False, "Audio file contains no samples"
                        
                        if sample_rate < 8000 or sample_rate > 96000:
                            return False, f"Invalid sample rate: {sample_rate}Hz"
                        
                        if channels < 1 or channels > 2:
                            return False, f"Invalid channel count: {channels}"
                        
                        if sample_width < 1 or sample_width > 4:
                            return False, f"Invalid sample width: {sample_width} bytes"
                        
                        # Check duration - more lenient thresholds for smart audio system
                        duration_ms = (frames / sample_rate) * 1000
                        min_duration = self.config.get("audio", {}).get("min_audio_length_ms", 100)
                        max_duration = self.config.get("audio", {}).get("max_audio_length_ms", 30000)
                        
                        if duration_ms < min_duration:
                            return False, f"Audio too short: {duration_ms:.1f}ms (min: {min_duration}ms)"
                        
                        if duration_ms > max_duration:
                            return False, f"Audio too long: {duration_ms:.1f}ms (max: {max_duration}ms)"
                        
                        # Enhanced audio quality check for smart system
                        pcm_data = wav_file.readframes(frames)
                        if sample_width == 2:  # 16-bit audio
                            samples = [int.from_bytes(pcm_data[i:i+2], byteorder='little', signed=True) 
                                     for i in range(0, len(pcm_data), 2)]
                            
                            # Calculate audio quality metrics
                            non_zero_samples = sum(1 for s in samples if abs(s) > 50)
                            significant_samples = sum(1 for s in samples if abs(s) > 500)
                            total_energy = sum(abs(s) for s in samples)
                            avg_energy = total_energy / len(samples) if samples else 0
                            
                            # Quality thresholds (more lenient for smart system)
                            non_zero_ratio = non_zero_samples / len(samples) if samples else 0
                            significant_ratio = significant_samples / len(samples) if samples else 0
                            
                            # Smart audio system should already filter quality, so be more permissive
                            if avg_energy < 10 and non_zero_ratio < 0.1:
                                return False, f"Audio quality too low: avg_energy={avg_energy:.1f}, non_zero_ratio={non_zero_ratio:.3f}"
                        
                        logger.debug(f"Audio validation passed: {frames} frames, {sample_rate}Hz, {channels}ch, {sample_width}B, {duration_ms:.1f}ms")
                        return True, f"Valid audio: {duration_ms:.1f}ms"
                        
                except wave.Error as e:
                    return False, f"WAV file corrupted: {e}"
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                return False, f"Audio validation failed: {e}"
                
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False, f"Validation error: {e}"
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using OpenAI Whisper with enhanced error handling"""
        try:
            if self.model is None:
                logger.error("Whisper model not loaded")
                return ""
            
            logger.info(f"Starting Whisper transcription of {len(audio_data)} bytes")
            
            # Validate audio before processing
            is_valid, validation_message = self._validate_audio(audio_data)
            if not is_valid:
                logger.warning(f"Audio validation failed: {validation_message}")
                return ""
            
            logger.debug(f"Audio validation: {validation_message}")
            
            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            try:
                logger.info(f"Created temporary audio file: {temp_audio_path}")
                
                # Run transcription in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                def whisper_transcribe():
                    try:
                        logger.info("Starting Whisper model transcription...")
                        
                        # Get Whisper configuration
                        whisper_opts = {
                            "language": self.language if self.language != "auto" else None,
                            "fp16": self.whisper_config.get("fp16", False),
                            "verbose": False,
                            "temperature": self.whisper_config.get("temperature", 0.0),
                            "condition_on_previous_text": self.whisper_config.get("condition_on_previous_text", False),
                            "no_speech_threshold": self.whisper_config.get("no_speech_threshold", 0.6),
                            "logprob_threshold": self.whisper_config.get("logprob_threshold", -1.0),
                            "compression_ratio_threshold": self.whisper_config.get("compression_ratio_threshold", 2.4),
                            "beam_size": self.whisper_config.get("beam_size", 1),
                            "best_of": self.whisper_config.get("best_of", 1),
                            "patience": self.whisper_config.get("patience", 1.0),
                            "suppress_blank": self.whisper_config.get("suppress_blank", True),
                            "word_timestamps": self.whisper_config.get("word_timestamps", False)
                        }
                        
                        # Add optional parameters if they exist
                        if "suppress_tokens" in self.whisper_config:
                            whisper_opts["suppress_tokens"] = self.whisper_config["suppress_tokens"]
                        
                        if "initial_prompt" in self.whisper_config and self.whisper_config["initial_prompt"]:
                            whisper_opts["initial_prompt"] = self.whisper_config["initial_prompt"]
                        
                        if "prepend_punctuations" in self.whisper_config:
                            whisper_opts["prepend_punctuations"] = self.whisper_config["prepend_punctuations"]
                        
                        if "append_punctuations" in self.whisper_config:
                            whisper_opts["append_punctuations"] = self.whisper_config["append_punctuations"]
                        
                        logger.debug(f"Whisper options: {whisper_opts}")
                        
                        result = self.model.transcribe(temp_audio_path, **whisper_opts)
                        logger.info("Whisper model transcription completed")
                        return result
                        
                    except Exception as e:
                        logger.error(f"Whisper transcription error in thread: {e}")
                        # Check for specific error types
                        error_msg = str(e).lower()
                        if "reshape tensor" in error_msg or "tensor of 0 elements" in error_msg:
                            logger.error("Whisper tensor reshape error - likely empty/corrupted audio")
                        elif "out of memory" in error_msg or "cuda" in error_msg:
                            logger.error("Whisper memory/CUDA error - try reducing batch size")
                        elif "timeout" in error_msg:
                            logger.error("Whisper internal timeout")
                        raise
                
                # Add timeout protection for the transcription itself
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, whisper_transcribe),
                    timeout=self.timeout_seconds
                )
                
                transcript = result.get("text", "").strip()
                logger.info(f"Whisper transcription result: '{transcript}' (length: {len(transcript)})")
                return transcript
                
            except asyncio.TimeoutError:
                logger.error(f"Whisper transcription timed out after {self.timeout_seconds} seconds")
                return ""
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
                return ""
            finally:
                # Clean up temporary file - CRITICAL for privacy
                if os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                        logger.debug(f"Cleaned up temporary file: {temp_audio_path}")
                    except Exception as e:
                        logger.error(f"SECURITY: Failed to cleanup temp file {temp_audio_path}: {e}")
                        # Try alternative cleanup methods
                        try:
                            os.chmod(temp_audio_path, 0o666)
                            os.unlink(temp_audio_path)
                            logger.info(f"Alternative cleanup successful: {temp_audio_path}")
                        except Exception as e2:
                            logger.critical(f"SECURITY: Unable to delete audio file {temp_audio_path}: {e2}")
                    
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return ""
    
    async def cleanup(self):
        """Clean up Whisper resources"""
        self.model = None
        logger.info("Whisper transcriber cleanup completed")

class VoskTranscriber(BaseTranscriber):
    """Vosk based transcriber"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.rec = None
        self._initialize_vosk()
        
        logger.info(f"Initialized Vosk transcriber with model: {model_path}")
    
    def _initialize_vosk(self):
        """Initialize Vosk model and recognizer"""
        try:
            import vosk
            import json
            
            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Vosk model not found at: {self.model_path}")
            
            self.model = vosk.Model(self.model_path)
            self.rec = vosk.KaldiRecognizer(self.model, 16000)  # 16kHz sample rate
            
            logger.info("Vosk model loaded successfully")
            
        except ImportError:
            logger.error("Vosk not installed. Install with: pip install vosk")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using Vosk"""
        try:
            if self.rec is None:
                logger.error("Vosk recognizer not initialized")
                return ""
            
            # Process audio data (assuming 16-bit PCM at 16kHz)
            self.rec.AcceptWaveform(audio_data)
            result = self.rec.FinalResult()
            
            # Parse result
            result_dict = json.loads(result)
            transcript = result_dict.get("text", "").strip()
            
            return transcript
            
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return ""
    
    async def cleanup(self):
        """Clean up Vosk resources"""
        logger.info("Vosk transcriber cleanup completed")

def create_transcriber(transcriber_type: str, **kwargs) -> BaseTranscriber:
    """Factory function to create transcriber instances"""
    if transcriber_type == "whisper":
        return WhisperTranscriber(**kwargs)
    elif transcriber_type == "vosk":
        return VoskTranscriber(**kwargs)
    else:
        raise ValueError(f"Unknown transcriber type: {transcriber_type}") 