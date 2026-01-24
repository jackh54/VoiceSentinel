import asyncio
import logging
import json
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMProfanityDetector:
    def __init__(self, config: dict):
        self.config = config.get("llm_profanity", {})
        self.enabled = self.config.get("enabled", False)
        self.provider = LLMProvider(self.config.get("provider", "openai"))
        self.api_key = self.config.get("api_key", "")
        self.model = self.config.get("model", "gpt-3.5-turbo")
        default_timeout = 15 if self.provider == LLMProvider.OLLAMA else 5
        self.timeout_seconds = self.config.get("timeout_seconds", default_timeout)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.strictness = self.config.get("strictness", "medium")
        self.max_concurrent = self.config.get("max_concurrent_requests", 3)
        self.fallback_on_error = self.config.get("fallback_on_error", True)
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.client = None
        
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        try:
            if self.provider == LLMProvider.OPENAI:
                try:
                    from openai import AsyncOpenAI
                    self.client = AsyncOpenAI(api_key=self.api_key)
                    logger.info("LLM Profanity Detector initialized with OpenAI")
                except ImportError:
                    logger.error("OpenAI library not installed. Install with: pip install openai")
                    self.enabled = False
            
            elif self.provider == LLMProvider.ANTHROPIC:
                try:
                    from anthropic import AsyncAnthropic
                    self.client = AsyncAnthropic(api_key=self.api_key)
                    logger.info("LLM Profanity Detector initialized with Anthropic Claude")
                except ImportError:
                    logger.error("Anthropic library not installed. Install with: pip install anthropic")
                    self.enabled = False
            
            elif self.provider == LLMProvider.OLLAMA:
                try:
                    from openai import AsyncOpenAI
                    if self.api_key:
                        base_url = self.api_key.rstrip('/')
                        if not base_url.endswith('/v1'):
                            base_url = f"{base_url}/v1"
                    else:
                        base_url = "http://localhost:11434/v1"
                    self.client = AsyncOpenAI(
                        base_url=base_url,
                        api_key="ollama"
                    )
                    logger.info(f"LLM Profanity Detector initialized with Ollama at {base_url}")
                except ImportError:
                    logger.error("OpenAI library not installed. Install with: pip install openai")
                    self.enabled = False
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False
    
    async def check_profanity(self, transcript: str, detected_language: str = "en") -> Tuple[bool, float, str]:
        if not self.enabled or not self.client or not transcript:
            return False, 0.0, "LLM disabled or not configured"
        
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    self._check_with_provider(transcript, detected_language),
                    timeout=self.timeout_seconds
                )
                return result
            
            except asyncio.TimeoutError:
                logger.warning(f"LLM profanity check timeout after {self.timeout_seconds}s for transcript: '{transcript[:50]}...'")
                if not self.fallback_on_error:
                    return False, 0.0, "Timeout"
                return self._fallback_result()
            
            except Exception as e:
                logger.error(f"LLM profanity check failed: {e}", exc_info=True)
                if not self.fallback_on_error:
                    return False, 0.0, f"Error: {str(e)}"
                return self._fallback_result()
    
    async def _check_with_provider(self, transcript: str, detected_language: str) -> Tuple[bool, float, str]:
        if self.provider == LLMProvider.OPENAI or self.provider == LLMProvider.OLLAMA:
            return await self._check_openai(transcript, detected_language)
        elif self.provider == LLMProvider.ANTHROPIC:
            return await self._check_anthropic(transcript, detected_language)
        else:
            return False, 0.0, "Unknown provider"
    
    async def _check_openai(self, transcript: str, detected_language: str) -> Tuple[bool, float, str]:
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(transcript, detected_language)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.0,
                max_tokens=50
            )
            
            if not response.choices or not response.choices[0].message:
                logger.warning("LLM returned empty response")
                return False, 0.0, "Empty response"
            
            content = response.choices[0].message.content.strip()
            if not content:
                logger.warning("LLM returned empty content")
                return False, 0.0, "Empty content"
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                error_msg = str(e) if hasattr(e, 'msg') else (content[:100] if content else "<no content available>")
                logger.warning(f"Failed to parse LLM response as JSON: {error_msg}")
                return False, 0.0, "Invalid JSON response"
            
            is_profane = result.get("is_profane", False)
            try:
                confidence = float(result.get("confidence", 0.0))
            except (ValueError, TypeError):
                logger.warning(f"LLM returned invalid confidence value: {result.get('confidence')}, defaulting to 0.0")
                confidence = 0.0
            reason = result.get("reason", "No reason provided")
            
            flagged = is_profane and confidence >= self.confidence_threshold
            
            return flagged, confidence, reason
        
        except Exception as e:
            logger.error(f"LLM API call failed: {e}", exc_info=True)
            raise
    
    async def _check_anthropic(self, transcript: str, detected_language: str) -> Tuple[bool, float, str]:
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(transcript, detected_language)
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=50,
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            if not hasattr(response, 'content') or not response.content:
                logger.warning("LLM response missing content attribute or content is empty")
                return False, 0.0, "Empty response content"
            
            if not isinstance(response.content, list) or len(response.content) == 0:
                logger.warning("LLM response content is not a non-empty list")
                return False, 0.0, "Invalid response content format"
            
            if not hasattr(response.content[0], 'text') or not response.content[0].text:
                logger.warning("LLM response content[0] missing text attribute or text is empty")
                return False, 0.0, "Empty response text"
            
            content = response.content[0].text.strip()
            if not content:
                logger.warning("LLM response text is empty after stripping")
                return False, 0.0, "Empty response text"
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                error_msg = str(e) if hasattr(e, 'msg') else "<no content available>"
                logger.warning(f"Failed to parse LLM response as JSON: {error_msg}")
                return False, 0.0, "Invalid JSON response"
            
            is_profane = result.get("is_profane", False)
            try:
                confidence = float(result.get("confidence", 0.0))
            except (ValueError, TypeError):
                logger.warning(f"LLM returned invalid confidence value: {result.get('confidence')}, defaulting to 0.0")
                confidence = 0.0
            reason = result.get("reason", "No reason provided")
            
            flagged = is_profane and confidence >= self.confidence_threshold
            
            return flagged, confidence, reason
        
        except json.JSONDecodeError as e:
            error_msg = str(e) if hasattr(e, 'msg') else "<no content available>"
            logger.warning(f"Failed to parse LLM response as JSON: {error_msg}")
            return False, 0.0, "Invalid JSON response"
        except Exception as e:
            raise
    
    def _get_system_prompt(self) -> str:
        strictness_instructions = {
            "strict": "Only flag severe profanity, hate speech, or threats. Ignore mild language.",
            "medium": "Flag profanity, hate speech, threats, and clearly inappropriate content.",
            "lenient": "Flag only obvious profanity and serious violations."
        }
        instruction = strictness_instructions.get(self.strictness, strictness_instructions["medium"])
        return f"Moderate content. {instruction} Respond ONLY with JSON: {{\"is_profane\":bool,\"confidence\":0.0-1.0,\"reason\":\"brief\"}}"
    
    def _build_prompt(self, transcript: str, detected_language: str) -> str:
        return f'Text: "{transcript}"'
    
    def _get_language_name(self, code: str) -> str:
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        return language_names.get(code, code.upper())
    
    def _fallback_result(self) -> Tuple[bool, float, str]:
        return False, 0.0, "Fallback: no detection"
