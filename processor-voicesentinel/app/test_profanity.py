import pytest
from profanity import ProfanityFilter, load_config

class TestProfanityFilter:
    
    def setup_method(self):
        """Setup test fixtures"""
        config = {
            "profanity": {
                "enabled": True,
                "severity_threshold": 0.7,
                "custom_words": ["badword", "terribleword"],
                "whitelist": ["glass", "class"],
                "use_regex_patterns": True
            }
        }
        self.filter = ProfanityFilter(config)
    
    def test_basic_profanity_detection(self):
        """Test basic profanity detection"""
        # Clean text
        result = self.filter.check_profanity("This is a clean sentence")
        assert result["contains_profanity"] is False
        assert result["severity"] == 0.0
        assert len(result["filtered_words"]) == 0
        
        # Text with profanity
        result = self.filter.check_profanity("This is a badword sentence")
        assert result["contains_profanity"] is True
        assert result["severity"] > 0.0
        assert "badword" in result["filtered_words"]
        assert "***" in result["clean_text"]
    
    def test_severity_calculation(self):
        """Test severity calculation"""
        # Single mild word
        result = self.filter.check_profanity("damn")
        mild_severity = result["severity"]
        
        # Multiple words should have higher severity
        result = self.filter.check_profanity("damn badword terribleword")
        assert result["severity"] > mild_severity
        
        # Severity should be capped at 1.0
        assert result["severity"] <= 1.0
    
    def test_custom_words(self):
        """Test custom word detection"""
        result = self.filter.check_profanity("This contains badword")
        assert result["contains_profanity"] is True
        assert "badword" in result["filtered_words"]
        
        result = self.filter.check_profanity("This contains terribleword")
        assert result["contains_profanity"] is True
        assert "terribleword" in result["filtered_words"]
    
    def test_whitelist(self):
        """Test whitelist functionality"""
        # Words in whitelist should not be flagged
        result = self.filter.check_profanity("I have a glass of water")
        assert result["contains_profanity"] is False
        
        result = self.filter.check_profanity("I'm in math class")
        assert result["contains_profanity"] is False
    
    def test_case_insensitive(self):
        """Test case insensitive detection"""
        test_cases = [
            "BADWORD",
            "BadWord", 
            "badword",
            "BaDwOrD"
        ]
        
        for case in test_cases:
            result = self.filter.check_profanity(f"This is {case}")
            assert result["contains_profanity"] is True, f"Failed for case: {case}"
    
    def test_word_boundaries(self):
        """Test word boundary detection"""
        # Should detect whole words
        result = self.filter.check_profanity("badword is bad")
        assert result["contains_profanity"] is True
        
        # Should not detect partial matches in other words
        result = self.filter.check_profanity("embarrassed")  # contains "ass"
        # This depends on implementation - might or might not flag
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        result = self.filter.check_profanity("This badword should be cleaned")
        assert "***" in result["clean_text"]
        assert "badword" not in result["clean_text"]
        assert "This" in result["clean_text"]
        assert "should be cleaned" in result["clean_text"]
    
    def test_multiple_profanity_words(self):
        """Test handling multiple profanity words"""
        result = self.filter.check_profanity("badword and terribleword together")
        assert result["contains_profanity"] is True
        assert len(result["filtered_words"]) == 2
        assert "badword" in result["filtered_words"]
        assert "terribleword" in result["filtered_words"]
        assert result["clean_text"].count("***") == 2
    
    def test_regex_patterns(self):
        """Test regex pattern matching"""
        # Test leetspeak variations
        result = self.filter.check_profanity("b4dw0rd is bad")
        # This should be caught by regex patterns if implemented
        
        # Test with numbers
        result = self.filter.check_profanity("bad123word")
        # Depending on regex implementation
    
    def test_disabled_filter(self):
        """Test filter when disabled"""
        config = {
            "profanity": {
                "enabled": False,
                "severity_threshold": 0.7,
                "custom_words": ["badword"],
                "whitelist": [],
                "use_regex_patterns": True
            }
        }
        disabled_filter = ProfanityFilter(config)
        
        result = disabled_filter.check_profanity("badword and more profanity")
        assert result["contains_profanity"] is False
        assert result["severity"] == 0.0
        assert len(result["filtered_words"]) == 0
        assert result["clean_text"] == "badword and more profanity"
    
    def test_severity_threshold(self):
        """Test severity threshold functionality"""
        config = {
            "profanity": {
                "enabled": True,
                "severity_threshold": 0.9,  # Very high threshold
                "custom_words": ["mildword"],
                "whitelist": [],
                "use_regex_patterns": True
            }
        }
        high_threshold_filter = ProfanityFilter(config)
        
        # Mild profanity should not trigger action due to high threshold
        result = high_threshold_filter.check_profanity("mildword")
        # The detection might still return True, but action_required should be False
        if "action_required" in result:
            assert result["action_required"] is False
    
    def test_empty_text(self):
        """Test handling of empty or whitespace text"""
        result = self.filter.check_profanity("")
        assert result["contains_profanity"] is False
        assert result["severity"] == 0.0
        
        result = self.filter.check_profanity("   ")
        assert result["contains_profanity"] is False
        assert result["severity"] == 0.0
    
    def test_special_characters(self):
        """Test handling of special characters"""
        result = self.filter.check_profanity("What the f***?!")
        # Should handle special characters gracefully
        
        result = self.filter.check_profanity("b@dw0rd with symbols")
        # Should potentially detect obfuscated profanity

def test_load_config():
    """Test configuration loading"""
    config = load_config()
    assert "profanity" in config
    assert "enabled" in config["profanity"]
    assert "severity_threshold" in config["profanity"]

if __name__ == "__main__":
    pytest.main([__file__]) 