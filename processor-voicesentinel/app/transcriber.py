import asyncio
import io
import logging
import os
import tempfile
import wave

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TARGET_SAMPLE_RATE = 16000


def _decode_wav_bytes(audio_data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_data), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples, sample_rate


def _resample_to_target(audio: np.ndarray, sample_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if sample_rate == target_rate or len(audio) == 0:
        return audio
    ratio = sample_rate / target_rate
    indices = (np.arange(int(len(audio) / ratio)) * ratio).astype(np.int64)
    indices = indices[indices < len(audio)]
    return audio[indices]


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "Systran/faster-whisper-large-v3", language: str = "auto", config: dict = None):
        self.model_name = model_name
        self.language = language
        self.config = config or {}
        self.model = None
        self._load_model()
        transcription = self.config.get("transcription", {})
        self.timeout_seconds = transcription.get("timeout_seconds", 30)
        self.beam_size = max(1, int(transcription.get("beam_size", 5)))
        self.vad_filter = bool(transcription.get("vad_filter", True))
        self.use_memory_input = bool(transcription.get("use_memory_input", False))
        audio_cfg = self.config.get("audio", {})
        min_ms = int(audio_cfg.get("min_audio_length_ms", 50))
        sample_rate = int(audio_cfg.get("sample_rate", TARGET_SAMPLE_RATE))
        channels = int(audio_cfg.get("channels", 1))
        self._min_audio_bytes = 44 + int(min_ms * (sample_rate * channels * 2) / 1000.0)

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

    def _transcribe_sync(self, audio_data: bytes) -> tuple:
        language = None if self.language == "auto" else self.language
        transcribe_kwargs = {
            "language": language,
            "beam_size": self.beam_size,
            "temperature": 0.0,
            "vad_filter": self.vad_filter,
        }

        if self.use_memory_input:
            try:
                audio_array, sample_rate = _decode_wav_bytes(audio_data)
                audio_array = _resample_to_target(audio_array, sample_rate)
                segments, info = self.model.transcribe(audio_array, **transcribe_kwargs)
                return self._segments_to_result(segments, info)
            except Exception as e:
                logger.debug("In-memory transcribe failed, falling back to temp file: %s", e)

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            segments, info = self.model.transcribe(temp_audio_path, **transcribe_kwargs)
            return self._segments_to_result(segments, info)
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass

    @staticmethod
    def _segments_to_result(segments, info) -> tuple:
        transcript = "\n".join(
            seg.text.strip() for seg in segments if seg.text and seg.text.strip()
        ).strip()
        raw = getattr(info, "language", None) if info is not None else None
        if isinstance(raw, str) and raw.strip():
            detected_language = raw.strip()
        else:
            detected_language = "unknown"
        return transcript, detected_language

    async def transcribe(self, audio_data: bytes) -> tuple:
        if not self.model or len(audio_data) < self._min_audio_bytes:
            return "", "unknown"

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._transcribe_sync, audio_data),
                timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            logger.error("Transcription timeout")
            return "", "unknown"
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", "unknown"


def create_transcriber(transcriber_type: str, **kwargs):
    return FasterWhisperTranscriber(**kwargs)
