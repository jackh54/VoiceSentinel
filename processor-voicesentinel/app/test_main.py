import pytest
import asyncio
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from main import app, TranscribeRequest, TranscribeResponse

# Test client
client = TestClient(app)

class TestProcessorEndpoints:
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    @patch('main.transcriber')
    @patch('main.decoder')
    def test_transcribe_endpoint_success(self, mock_decoder, mock_transcriber):
        """Test successful transcription"""
        # Mock audio data
        test_audio_data = b"fake_wav_audio_data"
        encoded_audio = base64.b64encode(test_audio_data).decode()
        
        # Mock decoder response
        mock_decoder.decode_opus_to_pcm.return_value = b"fake_pcm_data"
        
        # Mock transcriber response
        mock_transcriber.transcribe_audio.return_value = {
            "text": "hello world",
            "confidence": 0.95,
            "duration": 2.0
        }
        
        request_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": encoded_audio,
            "server_key": "test_key_1234567890123456"
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["transcription"]["text"] == "hello world"
        assert data["transcription"]["confidence"] == 0.95
        assert data["profanity_detected"] is False
    
    @patch('main.transcriber')
    @patch('main.decoder')
    @patch('main.profanity_filter')
    def test_transcribe_with_profanity(self, mock_profanity, mock_decoder, mock_transcriber):
        """Test transcription with profanity detection"""
        test_audio_data = b"fake_opus_audio_data"
        encoded_audio = base64.b64encode(test_audio_data).decode()
        
        mock_decoder.decode_opus_to_pcm.return_value = b"fake_pcm_data"
        mock_transcriber.transcribe_audio.return_value = {
            "text": "this is bad word",
            "confidence": 0.90,
            "duration": 1.5
        }
        
        # Mock profanity detection
        mock_profanity.check_profanity.return_value = {
            "contains_profanity": True,
            "severity": 0.8,
            "filtered_words": ["bad"],
            "clean_text": "this is *** word"
        }
        
        request_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": encoded_audio,
            "server_key": "test_key_1234567890123456"
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["profanity_detected"] is True
        assert data["profanity_info"]["severity"] == 0.8
        assert "bad" in data["profanity_info"]["filtered_words"]
    
    def test_transcribe_invalid_audio_data(self):
        """Test transcription with invalid audio data"""
        request_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": "invalid_base64_data!@#",
            "server_key": "test_key_1234567890123456"
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_transcribe_missing_fields(self):
        """Test transcription with missing required fields"""
        request_data = {
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            # Missing audio_data, format, sample_rate
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('main.decoder')
    def test_transcribe_decoder_error(self, mock_decoder):
        """Test transcription when decoder fails"""
        test_audio_data = b"fake_opus_audio_data"
        encoded_audio = base64.b64encode(test_audio_data).decode()
        
        # Mock decoder to raise exception
        mock_decoder.decode_opus_to_pcm.side_effect = Exception("Decode failed")
        
        request_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": encoded_audio,
            "server_key": "test_key_1234567890123456"
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    @patch('main.transcriber')
    @patch('main.decoder')
    def test_transcribe_transcriber_error(self, mock_decoder, mock_transcriber):
        """Test transcription when transcriber fails"""
        test_audio_data = b"fake_opus_audio_data"
        encoded_audio = base64.b64encode(test_audio_data).decode()
        
        mock_decoder.decode_opus_to_pcm.return_value = b"fake_pcm_data"
        mock_transcriber.transcribe_audio.side_effect = Exception("Transcription failed")
        
        request_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": encoded_audio,
            "server_key": "test_key_1234567890123456"
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data

class TestModels:
    
    def test_transcribe_request_validation(self):
        """Test TranscribeRequest model validation"""
        # Valid request
        valid_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": "dGVzdA==",  # base64 for "test"
            "server_key": "test_key_1234567890123456"
        }
        
        request = TranscribeRequest(**valid_data)
        assert request.session_id == "test_session_12345678"
        assert request.player == "TestPlayer"
        assert request.audio_format == "wav"
    
    def test_transcribe_request_invalid_session_id(self):
        """Test TranscribeRequest with invalid session ID"""
        invalid_data = {
            "session_id": "short",  # Too short
            "player": "TestPlayer",
            "timestamp": 1234567890,
            "audio_format": "wav",
            "audio_data": "dGVzdA==",
            "server_key": "test_key_1234567890123456"
        }
        
        with pytest.raises(ValueError):
            TranscribeRequest(**invalid_data)
    
    def test_transcribe_response_creation(self):
        """Test TranscribeResponse model creation"""
        response_data = {
            "session_id": "test_session_12345678",
            "player": "TestPlayer",
            "flagged": False,
            "bad_words": [],
            "transcript": "hello world",
            "chunks_processed": 1,
            "processing_time_ms": 1500
        }
        
        response = TranscribeResponse(**response_data)
        assert response.session_id == "test_session_12345678"
        assert response.player == "TestPlayer"
        assert response.flagged is False
        assert response.bad_words == []
        assert response.transcript == "hello world"

if __name__ == "__main__":
    pytest.main([__file__]) 