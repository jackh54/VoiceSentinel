"""
Profanity filter module for detecting inappropriate content
"""

import logging
import re
from typing import List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for profanity filtering"""
    case_sensitive: bool = False
    partial_match: bool = True
    whitelist: List[str] = None
    severity_levels: bool = False

class ProfanityFilter:
    """Profanity filter for text content"""
    
    def __init__(self, words: List[str] = None, case_sensitive: bool = False, 
                 partial_match: bool = True, whitelist: List[str] = None):
        """
        Initialize profanity filter
        
        Args:
            words: List of words to filter
            case_sensitive: Whether filtering is case sensitive
            partial_match: Whether to match partial words
            whitelist: List of words to exclude from filtering
        """
        self.words = set(words or [])
        self.case_sensitive = case_sensitive
        self.partial_match = partial_match
        self.whitelist = set(whitelist or [])
        
        # Normalize words based on case sensitivity
        if not self.case_sensitive:
            self.words = {word.lower() for word in self.words}
            self.whitelist = {word.lower() for word in self.whitelist}
        
        # Compile regex patterns for efficient matching
        self.patterns = self._compile_patterns()
        
        logger.info(f"Initialized profanity filter with {len(self.words)} words")
        logger.info(f"Case sensitive: {case_sensitive}, Partial match: {partial_match}")
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for word matching"""
        patterns = []
        
        for word in self.words:
            if word in self.whitelist:
                continue
                
            # Escape special regex characters
            escaped_word = re.escape(word)
            
            if self.partial_match:
                # Match word anywhere in text
                pattern = escaped_word
            else:
                # Match whole words only
                pattern = r'\b' + escaped_word + r'\b'
            
            # Compile with appropriate flags
            flags = 0 if self.case_sensitive else re.IGNORECASE
            
            try:
                compiled_pattern = re.compile(pattern, flags)
                patterns.append(compiled_pattern)
            except re.error as e:
                logger.warning(f"Failed to compile pattern for word '{word}': {e}")
        
        return patterns
    
    def check_text(self, text: str) -> List[str]:
        """
        Check text for profanity
        
        Args:
            text: Text to check
            
        Returns:
            List of flagged words found in the text
        """
        if not text or not text.strip():
            return []
        
        flagged_words = []
        
        # Normalize text for matching
        check_text = text if self.case_sensitive else text.lower()
        
        # Check against each pattern
        for i, pattern in enumerate(self.patterns):
            matches = pattern.findall(check_text)
            
            if matches:
                # Get the original word that matched
                original_word = list(self.words)[i] if i < len(self.words) else "unknown"
                
                # Add to flagged words (avoid duplicates)
                if original_word not in flagged_words:
                    flagged_words.append(original_word)
        
        # Also do simple word-by-word checking as fallback
        words_in_text = self._extract_words(check_text)
        for word in words_in_text:
            if word in self.words and word not in self.whitelist:
                if word not in flagged_words:
                    flagged_words.append(word)
        
        if flagged_words:
            logger.info(f"Found {len(flagged_words)} flagged words in text: {flagged_words}")
        
        return flagged_words
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract individual words from text"""
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        return [word.lower() if not self.case_sensitive else word for word in words]
    
    def is_flagged(self, text: str) -> bool:
        """
        Check if text contains any flagged content
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains flagged content
        """
        return len(self.check_text(text)) > 0
    
    def add_words(self, words: List[str]):
        """Add words to the filter list"""
        for word in words:
            normalized_word = word.lower() if not self.case_sensitive else word
            self.words.add(normalized_word)
        
        # Recompile patterns
        self.patterns = self._compile_patterns()
        logger.info(f"Added {len(words)} words to filter")
    
    def remove_words(self, words: List[str]):
        """Remove words from the filter list"""
        for word in words:
            normalized_word = word.lower() if not self.case_sensitive else word
            self.words.discard(normalized_word)
        
        # Recompile patterns
        self.patterns = self._compile_patterns()
        logger.info(f"Removed {len(words)} words from filter")
    
    def add_to_whitelist(self, words: List[str]):
        """Add words to the whitelist (exclude from filtering)"""
        for word in words:
            normalized_word = word.lower() if not self.case_sensitive else word
            self.whitelist.add(normalized_word)
        
        # Recompile patterns
        self.patterns = self._compile_patterns()
        logger.info(f"Added {len(words)} words to whitelist")
    
    def get_stats(self) -> dict:
        """Get statistics about the filter"""
        return {
            "total_words": len(self.words),
            "whitelist_words": len(self.whitelist),
            "case_sensitive": self.case_sensitive,
            "partial_match": self.partial_match,
            "patterns_compiled": len(self.patterns)
        }

class AdvancedProfanityFilter(ProfanityFilter):
    """Advanced profanity filter with additional features"""
    
    def __init__(self, *args, **kwargs):
        # Extract additional config
        self.leetspeak_detection = kwargs.pop('leetspeak_detection', True)
        self.repeated_char_detection = kwargs.pop('repeated_char_detection', True)
        self.spacing_detection = kwargs.pop('spacing_detection', True)
        
        super().__init__(*args, **kwargs)
        
        # Leetspeak mapping
        self.leetspeak_map = {
            '4': 'a', '@': 'a', '3': 'e', '1': 'i', '!': 'i', 
            '0': 'o', '5': 's', '$': 's', '7': 't', '+': 't',
            '2': 'z', '8': 'b', '6': 'g', '9': 'g'
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for advanced detection"""
        normalized = text.lower()
        
        # Handle leetspeak
        if self.leetspeak_detection:
            for leet, normal in self.leetspeak_map.items():
                normalized = normalized.replace(leet, normal)
        
        # Handle repeated characters (e.g., "shiiit" -> "shit")
        if self.repeated_char_detection:
            normalized = re.sub(r'(.)\1{2,}', r'\1', normalized)
        
        # Handle spacing (e.g., "s h i t" -> "shit")
        if self.spacing_detection:
            normalized = re.sub(r'\s+', '', normalized)
        
        return normalized
    
    def check_text(self, text: str) -> List[str]:
        """Enhanced text checking with normalization"""
        if not text or not text.strip():
            return []
        
        # Run standard check
        flagged_words = super().check_text(text)
        
        # Run advanced check on normalized text
        normalized_text = self._normalize_text(text)
        advanced_flagged = super().check_text(normalized_text)
        
        # Combine results
        all_flagged = list(set(flagged_words + advanced_flagged))
        
        return all_flagged

def create_default_filter() -> ProfanityFilter:
    """Create a default profanity filter with common words"""
    default_words = [
        # Basic inappropriate words (keeping minimal for configuration)
        "badword1", "badword2", "inappropriate", "offensive",
        # Placeholder entries - admins should replace with actual words
        "example1", "example2", "test1", "test2"
    ]
    
    return ProfanityFilter(
        words=default_words,
        case_sensitive=False,
        partial_match=True
    )

def load_filter_from_file(file_path: str) -> ProfanityFilter:
    """Load profanity filter from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        return ProfanityFilter(words=words)
        
    except FileNotFoundError:
        logger.warning(f"Profanity filter file not found: {file_path}")
        return create_default_filter()
    except Exception as e:
        logger.error(f"Failed to load profanity filter from {file_path}: {e}")
        return create_default_filter() 