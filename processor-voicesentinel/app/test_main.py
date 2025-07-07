import pytest
import asyncio
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from main import app, ProcessingRequest, ProcessingResponse

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
    
    def test_info_endpoint(self):
        """Test the info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "VoiceSentinel Processor"
        assert "transcription_engine" in data
        assert "supported_formats" in data
    
    @patch('main.transcriber')
    @patch('main.decoder')
    def test_transcribe_endpoint_success(self, mock_decoder, mock_transcriber):
        """Test successful transcription"""
        # Mock audio data
        test_audio_data = b"fake_opus_audio_data"
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
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": encoded_audio,
            "format": "opus",
            "sample_rate": 48000
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
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": encoded_audio,
            "format": "opus",
            "sample_rate": 48000
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
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": "invalid_base64_data!@#",
            "format": "opus",
            "sample_rate": 48000
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
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": encoded_audio,
            "format": "opus",
            "sample_rate": 48000
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
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": encoded_audio,
            "format": "opus",
            "sample_rate": 48000
        }
        
        response = client.post("/transcribe", json=request_data)
        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "error" in data

class TestModels:
    
    def test_processing_request_validation(self):
        """Test ProcessingRequest model validation"""
        # Valid request
        valid_data = {
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "audio_data": "dGVzdA==",  # base64 for "test"
            "format": "opus",
            "sample_rate": 48000
        }
        
        request = ProcessingRequest(**valid_data)
        assert str(request.player_id) == "550e8400-e29b-41d4-a716-446655440000"
        assert request.format == "opus"
        assert request.sample_rate == 48000
    
    def test_processing_request_invalid_uuid(self):
        """Test ProcessingRequest with invalid UUID"""
        invalid_data = {
            "player_id": "not-a-uuid",
            "audio_data": "dGVzdA==",
            "format": "opus",
            "sample_rate": 48000
        }
        
        with pytest.raises(ValueError):
            ProcessingRequest(**invalid_data)
    
    def test_processing_response_creation(self):
        """Test ProcessingResponse model creation"""
        response_data = {
            "success": True,
            "player_id": "550e8400-e29b-41d4-a716-446655440000",
            "transcription": {
                "text": "hello world",
                "confidence": 0.95,
                "duration": 2.0
            },
            "profanity_detected": False,
            "processing_time": 1.5
        }
        
        response = ProcessingResponse(**response_data)
        assert response.success is True
        assert response.transcription["text"] == "hello world"
        assert response.profanity_detected is False

if __name__ == "__main__":
    pytest.main([__file__]) 