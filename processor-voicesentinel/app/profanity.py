import re
from typing import List, Tuple, Dict

class ProfanityFilter:
    def __init__(self, words: List[str] = None, case_sensitive: bool = False, 
                 partial_match: bool = True, whitelist: List[str] = None):
        self.words = set(words or [])
        self.case_sensitive = case_sensitive
        self.partial_match = partial_match
        self.whitelist = set(whitelist or [])
        
        if not self.case_sensitive:
            self.words = {word.lower() for word in self.words}
            self.whitelist = {word.lower() for word in self.whitelist}
        
        self.patterns, self.pattern_to_word = self._compile_patterns()
    
    def _compile_patterns(self) -> Tuple[List[re.Pattern], Dict[re.Pattern, str]]:
        patterns = []
        pattern_to_word = {}
        for word in self.words:
            if word in self.whitelist:
                continue
            escaped_word = re.escape(word)
            pattern = escaped_word if self.partial_match else r'\b' + escaped_word + r'\b'
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                compiled_pattern = re.compile(pattern, flags)
                patterns.append(compiled_pattern)
                pattern_to_word[compiled_pattern] = word
            except:
                pass
        return patterns, pattern_to_word
    
    def check_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        flagged_words = []
        check_text = text if self.case_sensitive else text.lower()
        
        # Check patterns
        for pattern in self.patterns:
            if pattern.search(check_text):
                word = self.pattern_to_word.get(pattern)
                if word and word not in flagged_words:
                    flagged_words.append(word)
        
        # Fallback word-by-word check
        words_in_text = re.findall(r'\b\w+\b', check_text)
        normalized_words = [w.lower() if not self.case_sensitive else w for w in words_in_text]
        for word in normalized_words:
            if word in self.words and word not in self.whitelist and word not in flagged_words:
                flagged_words.append(word)
        
        return flagged_words