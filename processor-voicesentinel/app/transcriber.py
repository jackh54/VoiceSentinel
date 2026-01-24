import asyncio
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "Systran/faster-whisper-large-v3", language: str = "auto", config: dict = None):
        self.model_name = model_name
        self.language = language
        self.config = config or {}
        self.model = None
        self._load_model()
        self.timeout_seconds = self.config.get("transcription", {}).get("timeout_seconds", 30)
    
    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            import os
            
            # Force progress bars to show
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
            os.environ['TQDM_DISABLE'] = '0'
            
            # Set HuggingFace token if provided
            hf_token = self.config.get("transcription", {}).get("huggingface_token", "")
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                print(f"Using HuggingFace token for faster downloads", flush=True)
            
            device = self.config.get("transcription", {}).get("device", "cpu")
            compute_type = self.config.get("transcription", {}).get("compute_type", "int8")
            cpu_threads = self.config.get("transcription", {}).get("cpu_threads", 2)
            
            print(f"\n{'='*60}", flush=True)
            print(f"Loading Faster Whisper Model: {self.model_name}", flush=True)
            print(f"Device: {device} | Compute: {compute_type}", flush=True)
            print(f"This may take 5-15 minutes on first run (downloading ~3GB)", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            # Enable all huggingface logging and progress
            for log_name in ['huggingface_hub', 'huggingface_hub.file_download', 'tqdm']:
                logging.getLogger(log_name).setLevel(logging.INFO)
                logging.getLogger(log_name).propagate = True
            
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                num_workers=cpu_threads
            )
            
            print(f"\n{'='*60}", flush=True)
            print(f"Model loaded successfully!", flush=True)
            print(f"{'='*60}\n", flush=True)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    
    async def transcribe(self, audio_data: bytes) -> tuple:
        if not self.model or len(audio_data) < 5000:
            return "", "unknown"
        
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            loop = asyncio.get_event_loop()
            
            def transcribe_sync():
                segments, info = self.model.transcribe(
                    temp_audio_path,
                    language=None if self.language == "auto" else self.language,
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=True
                )
                transcript = " ".join([seg.text.strip() for seg in segments]).strip()
                detected_language = info.language if hasattr(info, 'language') else "unknown"
                return transcript, detected_language
            
            result = await asyncio.wait_for(
                loop.run_in_executor(None, transcribe_sync),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error("Transcription timeout")
            return "", "unknown"
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", "unknown"
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass




def create_transcriber(transcriber_type: str, **kwargs):
    return FasterWhisperTranscriber(**kwargs)
