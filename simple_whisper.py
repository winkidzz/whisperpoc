#!/usr/bin/env python3
"""
Simple Continuous Voice-to-Text using OpenAI Whisper
A simplified version for basic continuous transcription
"""

import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import time

class SimpleWhisperTranscriber:
    def __init__(self, model_name="base", sample_rate=16000, record_duration=5):
        """
        Initialize the simple Whisper transcriber
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            sample_rate: Audio sample rate in Hz
            record_duration: Duration of each recording segment in seconds
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        
        # Initialize Whisper model
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")
        
        # Recording state
        self.is_recording = False
        self.audio_data = []
        
    def record_audio(self):
        """Record audio for the specified duration"""
        print(f"🎤 Recording {self.record_duration} seconds...")
        
        # Record audio
        audio = sd.rec(
            int(self.record_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        return audio.flatten()
    
    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        try:
            result = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
                fp16=False
            )
            
            transcription = result["text"].strip()
            if transcription:
                print(f"🎤 {transcription}")
            else:
                print("🔇 No speech detected")
                
        except Exception as e:
            print(f"Transcription error: {e}")
    
    def start_continuous(self):
        """Start continuous recording and transcription"""
        print("🎧 Starting continuous voice-to-text transcription...")
        print("💡 Speak clearly into your microphone")
        print("⏹️  Press Ctrl+C to stop\n")
        
        self.is_recording = True
        
        try:
            while self.is_recording:
                # Record audio segment
                audio = self.record_audio()
                
                # Transcribe
                self.transcribe_audio(audio)
                
                print("-" * 50)
                
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
    print("🤖 Simple Whisper Live Transcriber")
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
    
    # Duration selection
    while True:
        try:
            duration = input(f"\nEnter recording duration in seconds (default=5): ").strip()
            if not duration:
                duration = 5
            else:
                duration = int(duration)
            
            if duration > 0:
                break
            else:
                print("Duration must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize and start transcriber
    transcriber = SimpleWhisperTranscriber(
        model_name=model_name,
        record_duration=duration
    )
    transcriber.start_continuous()

if __name__ == "__main__":
    main() 