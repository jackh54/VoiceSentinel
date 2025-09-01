#!/usr/bin/env python3
"""
WebSocket Test Client for VoiceSentinel Processor
Tests WebSocket connection, audio upload, and transcription reception
"""

import asyncio
import websockets
import json
import base64
import wave
import struct
import time
import argparse
from pathlib import Path

class VoiceSentinelTestClient:
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.websocket_url = f"ws://{host}:{port}/ws/test_client_{int(time.time())}"
        self.websocket = None
        
    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            print(f"🔗 Connecting to {self.websocket_url}...")
            self.websocket = await websockets.connect(self.websocket_url)
            print("✅ Connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            print("🔌 Disconnected")
    
    def generate_test_audio(self, duration=2, sample_rate=16000, frequency=440):
        """Generate a test audio WAV file in memory"""
        print(f"🎵 Generating {duration}s test audio at {frequency}Hz...")
        
        # Generate sine wave
        frames = int(duration * sample_rate)
        audio_data = []
        
        for i in range(frames):
            # Generate sine wave
            value = int(32767 * 0.3 * 
                       (0.5 * (1 + (i % (sample_rate // 10)) / (sample_rate // 10))))  # Varying amplitude
            audio_data.append(struct.pack('<h', value))
        
        # Create WAV file in memory
        wav_data = b''
        
        # WAV header
        wav_data += b'RIFF'
        wav_data += struct.pack('<I', 36 + len(audio_data) * 2)
        wav_data += b'WAVE'
        wav_data += b'fmt '
        wav_data += struct.pack('<I', 16)  # PCM format chunk size
        wav_data += struct.pack('<H', 1)   # PCM format
        wav_data += struct.pack('<H', 1)   # Mono
        wav_data += struct.pack('<I', sample_rate)
        wav_data += struct.pack('<I', sample_rate * 2)  # Byte rate
        wav_data += struct.pack('<H', 2)   # Block align
        wav_data += struct.pack('<H', 16)  # Bits per sample
        wav_data += b'data'
        wav_data += struct.pack('<I', len(audio_data) * 2)
        
        # Audio data
        for sample in audio_data:
            wav_data += sample
        
        print(f"📊 Generated WAV file: {len(wav_data)} bytes")
        return wav_data
    
    def load_audio_file(self, file_path):
        """Load audio file from disk"""
        try:
            print(f"📁 Loading audio file: {file_path}")
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            print(f"📊 Loaded audio file: {len(audio_data)} bytes")
            return audio_data
        except Exception as e:
            print(f"❌ Failed to load audio file: {e}")
            return None
    
    async def send_audio(self, audio_data, profanity_words=None):
        """Send audio data via WebSocket"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return False
        
        if profanity_words is None:
            profanity_words = ["test", "bad", "profanity"]
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Create message
            message = {
                "type": "audio_chunk",
                "audio_data": audio_b64,
                "profanity_words": profanity_words
            }
            
            print(f"📤 Sending audio chunk ({len(audio_data)} bytes)...")
            await self.websocket.send(json.dumps(message))
            print("✅ Audio sent successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
            return False
    
    async def listen_for_responses(self, timeout=30):
        """Listen for responses from the server"""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        print(f"👂 Listening for responses (timeout: {timeout}s)...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    # Parse response
                    try:
                        response = json.loads(message)
                        await self.handle_response(response)
                    except json.JSONDecodeError:
                        print(f"⚠️  Received non-JSON message: {message}")
                        
                except asyncio.TimeoutError:
                    # Continue listening
                    continue
                    
        except Exception as e:
            print(f"❌ Error while listening: {e}")
    
    async def handle_response(self, response):
        """Handle different types of responses"""
        msg_type = response.get("type", "unknown")
        
        if msg_type == "status":
            status = response.get("status", "unknown")
            print(f"📋 Status update: {status}")
            
        elif msg_type == "transcription_result":
            transcript = response.get("transcript", "")
            is_profane = response.get("is_profane", False)
            status = response.get("status", "unknown")
            
            print(f"📝 Transcription result:")
            print(f"   Text: '{transcript}'")
            print(f"   Profane: {'Yes' if is_profane else 'No'}")
            print(f"   Status: {status}")
            
        elif msg_type == "error":
            error_msg = response.get("message", "Unknown error")
            print(f"❌ Error from server: {error_msg}")
            
        else:
            print(f"❓ Unknown message type '{msg_type}': {response}")
    
    async def test_health_endpoint(self):
        """Test the health endpoint"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.host}:{self.port}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Health check passed: {data}")
                        return True
                    else:
                        print(f"❌ Health check failed: {response.status}")
                        return False
        except ImportError:
            print("⚠️  aiohttp not available, skipping health check")
            return True
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_stats_endpoint(self):
        """Test the stats endpoint"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{self.host}:{self.port}/stats") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"📊 Stats: {data}")
                        return True
                    else:
                        print(f"❌ Stats check failed: {response.status}")
                        return False
        except ImportError:
            print("⚠️  aiohttp not available, skipping stats check")
            return True
        except Exception as e:
            print(f"❌ Stats check error: {e}")
            return False

async def run_full_test(host="localhost", port=8000, audio_file=None):
    """Run a complete test of the WebSocket connection"""
    print("🚀 Starting VoiceSentinel WebSocket Test")
    print("=" * 50)
    
    client = VoiceSentinelTestClient(host, port)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    await client.test_health_endpoint()
    
    # Test 2: Stats check
    print("\n2️⃣ Testing stats endpoint...")
    await client.test_stats_endpoint()
    
    # Test 3: WebSocket connection
    print("\n3️⃣ Testing WebSocket connection...")
    if not await client.connect():
        print("❌ Test failed: Cannot connect to WebSocket")
        return False
    
    # Test 4: Audio processing
    print("\n4️⃣ Testing audio processing...")
    
    # Prepare audio data
    if audio_file and Path(audio_file).exists():
        audio_data = client.load_audio_file(audio_file)
    else:
        if audio_file:
            print(f"⚠️  Audio file {audio_file} not found, using generated audio")
        audio_data = client.generate_test_audio(duration=3)
    
    if audio_data is None:
        print("❌ Test failed: Cannot prepare audio data")
        await client.disconnect()
        return False
    
    # Start listening for responses
    listen_task = asyncio.create_task(client.listen_for_responses(timeout=20))
    
    # Wait a moment then send audio
    await asyncio.sleep(1)
    
    # Send audio with test profanity words
    test_profanity = ["test", "bad", "profanity", "damn"]
    success = await client.send_audio(audio_data, test_profanity)
    
    if not success:
        print("❌ Test failed: Cannot send audio")
        listen_task.cancel()
        await client.disconnect()
        return False
    
    # Wait for responses
    await listen_task
    
    # Test 5: Disconnect
    print("\n5️⃣ Testing disconnect...")
    await client.disconnect()
    
    print("\n🎉 Test completed!")
    return True

async def interactive_test(host="localhost", port=8000):
    """Run an interactive test session"""
    print("🎮 Interactive VoiceSentinel WebSocket Test")
    print("=" * 50)
    
    client = VoiceSentinelTestClient(host, port)
    
    if not await client.connect():
        print("❌ Cannot connect to WebSocket")
        return
    
    print("\nCommands:")
    print("  'gen' - Send generated test audio")
    print("  'file <path>' - Send audio file")
    print("  'quit' - Exit")
    
    # Start listening task
    listen_task = asyncio.create_task(client.listen_for_responses(timeout=300))
    
    try:
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'gen':
                    audio_data = client.generate_test_audio()
                    await client.send_audio(audio_data)
                elif command.startswith('file '):
                    file_path = command[5:].strip()
                    audio_data = client.load_audio_file(file_path)
                    if audio_data:
                        await client.send_audio(audio_data)
                else:
                    print("Unknown command. Use 'gen', 'file <path>', or 'quit'")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
                
    finally:
        listen_task.cancel()
        await client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Test VoiceSentinel WebSocket connection")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--audio", help="Path to audio file to test with")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        if args.interactive:
            asyncio.run(interactive_test(args.host, args.port))
        else:
            asyncio.run(run_full_test(args.host, args.port, args.audio))
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()
