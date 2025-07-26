#!/usr/bin/env python3
"""
Whisper Model Comparison Tool
Record audio and compare transcription accuracy across different Whisper models
"""

import whisper
import sounddevice as sd
import numpy as np
import time
import json
from datetime import datetime
import os

class WhisperModelComparator:
    def __init__(self, sample_rate=16000):
        """
        Initialize the model comparator
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.models = {}
        self.results = {}
        
    def load_models(self, model_names=None):
        """Load specified Whisper models"""
        if model_names is None:
            model_names = ['tiny', 'base', 'small']
        
        print("🤖 Loading Whisper models...")
        print("=" * 50)
        
        for model_name in model_names:
            print(f"📦 Loading {model_name} model...")
            start_time = time.time()
            
            try:
                model = whisper.load_model(model_name)
                load_time = time.time() - start_time
                
                self.models[model_name] = {
                    'model': model,
                    'load_time': load_time
                }
                
                print(f"✅ {model_name} loaded in {load_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        print(f"\n📊 Loaded {len(self.models)} models successfully")
    
    def record_audio(self, duration=10, filename=None):
        """
        Record audio for the specified duration
        
        Args:
            duration: Recording duration in seconds
            filename: Optional filename to save the audio
        """
        print(f"\n🎤 Recording {duration} seconds of audio...")
        print("💡 Speak clearly into your microphone")
        print("📹 Recording in progress...")
        
        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        # Show recording progress
        for i in range(duration):
            print(f"   Recording... {i+1}/{duration}")
            time.sleep(1)
        
        sd.wait()
        audio = audio.flatten()
        
        # Save audio if filename provided
        if filename:
            import scipy.io.wavfile as wav
            wav.write(filename, self.sample_rate, audio)
            print(f"💾 Audio saved to: {filename}")
        
        # Analyze audio
        max_amplitude = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        print(f"\n📊 Audio Analysis:")
        print(f"   Duration: {duration}s")
        print(f"   Max amplitude: {max_amplitude:.4f}")
        print(f"   RMS level: {rms:.4f}")
        
        if max_amplitude < 0.01:
            print("⚠️  Warning: Very low audio levels detected")
        
        return audio
    
    def transcribe_with_model(self, audio, model_name):
        """
        Transcribe audio using a specific model
        
        Args:
            audio: Audio data
            model_name: Name of the model to use
        """
        if model_name not in self.models:
            print(f"❌ Model {model_name} not loaded")
            return None
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        print(f"\n🎯 Transcribing with {model_name} model...")
        
        try:
            start_time = time.time()
            
            result = model.transcribe(
                audio,
                language="en",
                task="transcribe",
                fp16=False
            )
            
            transcription_time = time.time() - start_time
            transcription = result["text"].strip()
            
            return {
                'transcription': transcription,
                'transcription_time': transcription_time,
                'confidence': result.get('confidence', 0.0),
                'language': result.get('language', 'en')
            }
            
        except Exception as e:
            print(f"❌ Transcription failed with {model_name}: {e}")
            return None
    
    def compare_models(self, audio, save_results=True):
        """
        Compare transcription results across all loaded models
        
        Args:
            audio: Audio data to transcribe
            save_results: Whether to save results to file
        """
        print(f"\n🔍 Comparing {len(self.models)} models...")
        print("=" * 60)
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'audio_duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'models': {}
        }
        
        # Transcribe with each model
        for model_name in self.models.keys():
            result = self.transcribe_with_model(audio, model_name)
            
            if result:
                comparison_results['models'][model_name] = result
                
                print(f"\n📝 {model_name.upper()} Model Results:")
                print(f"   Transcription: '{result['transcription']}'")
                print(f"   Time: {result['transcription_time']:.2f}s")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Language: {result['language']}")
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"whisper_comparison_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        return comparison_results
    
    def print_summary(self, results):
        """Print a summary comparison of all models"""
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        models = list(results['models'].keys())
        
        if len(models) < 2:
            print("⚠️  Need at least 2 models for comparison")
            return
        
        print(f"Audio Duration: {results['audio_duration']:.1f}s")
        print(f"Sample Rate: {results['sample_rate']} Hz")
        print()
        
        # Print transcriptions side by side
        print("📝 TRANSCRIPTIONS:")
        print("-" * 60)
        
        max_length = max(len(model) for model in models)
        
        for model_name in models:
            result = results['models'][model_name]
            transcription = result['transcription']
            time_taken = result['transcription_time']
            
            print(f"{model_name.upper():<{max_length}} | {transcription}")
            print(f"{'':<{max_length}} | Time: {time_taken:.2f}s, Confidence: {result['confidence']:.3f}")
            print()
        
        # Speed comparison
        print("⚡ SPEED COMPARISON:")
        print("-" * 60)
        
        speeds = []
        for model_name in models:
            result = results['models'][model_name]
            speed = results['audio_duration'] / result['transcription_time']
            speeds.append((model_name, speed, result['transcription_time']))
        
        # Sort by speed (fastest first)
        speeds.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, speed, time_taken) in enumerate(speeds):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"{rank} {model_name.upper():<8} | {speed:.1f}x real-time | {time_taken:.2f}s")
        
        # Accuracy comparison (if confidence scores are available)
        print(f"\n🎯 CONFIDENCE COMPARISON:")
        print("-" * 60)
        
        confidences = []
        for model_name in models:
            result = results['models'][model_name]
            confidences.append((model_name, result['confidence']))
        
        # Sort by confidence (highest first)
        confidences.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, confidence) in enumerate(confidences):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"{rank} {model_name.upper():<8} | Confidence: {confidence:.3f}")

def main():
    """Main function"""
    print("🔬 Whisper Model Comparison Tool")
    print("=" * 50)
    
    # Initialize comparator
    comparator = WhisperModelComparator()
    
    # Load models
    comparator.load_models(['tiny', 'base', 'small'])
    
    if not comparator.models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Get recording duration
    while True:
        try:
            duration_input = input(f"\nEnter recording duration in seconds (default=10): ").strip()
            if not duration_input:
                duration = 10
            else:
                duration = int(duration_input)
            
            if duration > 0 and duration <= 60:
                break
            else:
                print("Duration must be between 1 and 60 seconds.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Record audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"test_audio_{timestamp}.wav"
    
    audio = comparator.record_audio(duration, audio_filename)
    
    # Compare models
    results = comparator.compare_models(audio)
    
    # Print summary
    comparator.print_summary(results)
    
    print(f"\n🎉 Comparison complete!")
    print(f"📁 Audio file: {audio_filename}")
    print(f"📊 Results saved to JSON file")

if __name__ == "__main__":
    main() 