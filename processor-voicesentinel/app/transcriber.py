import asyncio
import logging
import os
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

            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TQDM_DISABLE'] = '1'

            hf_token = self.config.get("transcription", {}).get("huggingface_token", "")
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                logger.info("Hugging Face token configured for model downloads")

            device = self.config.get("transcription", {}).get("device", "cpu")
            compute_type = self.config.get("transcription", {}).get("compute_type", "int8")
            cpu_threads = self.config.get("transcription", {}).get("cpu_threads", 2)

            logger.info(
                "Loading Faster Whisper model=%s device=%s compute=%s (first run may download several GB)",
                self.model_name,
                device,
                compute_type,
            )

            for log_name in ('huggingface_hub', 'huggingface_hub.file_download', 'tqdm'):
                logging.getLogger(log_name).setLevel(logging.WARNING)
                logging.getLogger(log_name).propagate = True
            
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                num_workers=cpu_threads
            )
            
            logger.info("Faster Whisper model loaded: %s", self.model_name)
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
            
            loop = asyncio.get_running_loop()
            
            def transcribe_sync():
                segments, info = self.model.transcribe(
                    temp_audio_path,
                    language=None if self.language == "auto" else self.language,
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=True
                )
                # One line per Whisper segment so the plugin buffer (and report books) can paginate each phrase.
                transcript = "\n".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip()).strip()
                raw = getattr(info, "language", None) if info is not None else None
                if isinstance(raw, str) and raw.strip():
                    detected_language = raw.strip()
                else:
                    detected_language = "unknown"
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
                except Exception:
                    pass




def create_transcriber(transcriber_type: str, **kwargs):
    return FasterWhisperTranscriber(**kwargs)
