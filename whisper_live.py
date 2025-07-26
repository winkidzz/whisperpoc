#!/usr/bin/env python3
"""
Continuous Voice-to-Text using OpenAI Whisper
Real-time speech recognition with continuous listening
"""

import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import webrtcvad
import collections
import sys
from typing import Optional

class WhisperLiveTranscriber:
    def __init__(self, model_name="base", sample_rate=16000, chunk_duration_ms=30):
        """
        Initialize the Whisper live transcriber
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in milliseconds
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # Initialize Whisper model
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        
        # Speech detection
        self.speech_buffer = collections.deque(maxlen=30)  # 30 chunks = 900ms
        self.silence_threshold = 0.8  # 80% silence to trigger transcription
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        
        # Transcription state
        self.current_audio = []
        self.last_transcription = ""
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        # Convert to int16 for VAD
        audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
        self.audio_queue.put(audio_chunk)
    
    def is_speech(self, audio_chunk):
        """Detect if audio chunk contains speech"""
        try:
            return self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)
        except:
            return False
    
    def process_audio(self):
        """Process audio chunks and detect speech segments"""
        while self.is_recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                is_speech = self.is_speech(audio_chunk)
                self.speech_buffer.append(is_speech)
                
                # Calculate speech ratio in recent chunks
                if len(self.speech_buffer) >= 10:
                    speech_ratio = sum(self.speech_buffer) / len(self.speech_buffer)
                    
                    if speech_ratio > 0.3:  # Speech detected
                        self.current_audio.append(audio_chunk)
                    elif len(self.current_audio) > 0:  # End of speech segment
                        # Check if we have enough audio to transcribe
                        audio_duration = len(self.current_audio) * self.chunk_duration_ms / 1000
                        if audio_duration >= self.min_speech_duration:
                            self.transcribe_audio()
                        self.current_audio = []
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def transcribe_audio(self):
        """Transcribe the current audio segment"""
        if not self.current_audio:
            return
        
        try:
            # Combine audio chunks
            audio_data = np.concatenate(self.current_audio)
            
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32767.0
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_float,
                language="en",
                task="transcribe",
                fp16=False
            )
            
            transcription = result["text"].strip()
            
            if transcription and transcription != self.last_transcription:
                print(f"\n🎤 {transcription}")
                self.last_transcription = transcription
                
        except Exception as e:
            print(f"Transcription error: {e}")
    
    def start(self):
        """Start continuous listening and transcription"""
        print("🎧 Starting continuous voice-to-text transcription...")
        print("💡 Speak clearly into your microphone")
        print("⏹️  Press Ctrl+C to stop\n")
        
        self.is_recording = True
        
        # Start audio processing thread
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
        
        try:
            # Start audio stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                print("🎤 Listening... (Press Ctrl+C to stop)")
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping transcription...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop transcription"""
        self.is_recording = False
        print("✅ Transcription stopped")

def main():
    """Main function"""
    print("🤖 Whisper Live Transcriber")
    print("=" * 40)
    
    # Model selection
    models = {
        "1": "tiny",
        "2": "base", 
        "3": "small",
        "4": "medium",
        "5": "large"
    }
    
    print("\nSelect Whisper model:")
    for key, model in models.items():
        print(f"  {key}. {model}")
    
    while True:
        choice = input("\nEnter your choice (1-5, default=2): ").strip()
        if not choice:
            choice = "2"
        
        if choice in models:
            model_name = models[choice]
            break
        else:
            print("Invalid choice. Please select 1-5.")
    
    # Initialize and start transcriber
    transcriber = WhisperLiveTranscriber(model_name=model_name)
    transcriber.start()

if __name__ == "__main__":
    main() 