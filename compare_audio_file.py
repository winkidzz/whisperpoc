#!/usr/bin/env python3
"""
Audio File Model Comparison Tool
Compare Whisper models on an existing audio file
"""

import whisper
import numpy as np
import time
import json
from datetime import datetime
import argparse
import os

def load_audio_file(file_path, target_sr=16000):
    """Load audio file and convert to the target sample rate"""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except ImportError:
        print("❌ librosa not available. Please install: pip install librosa")
        return None, None
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return None, None

def transcribe_with_model(audio, model, model_name):
    """Transcribe audio with a specific model"""
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

def compare_models_on_audio(audio, sample_rate, model_names=['tiny', 'base', 'small']):
    """Compare multiple models on the same audio"""
    print("🤖 Loading Whisper models...")
    print("=" * 50)
    
    models = {}
    
    # Load models
    for model_name in model_names:
        print(f"📦 Loading {model_name} model...")
        start_time = time.time()
        
        try:
            model = whisper.load_model(model_name)
            load_time = time.time() - start_time
            
            models[model_name] = {
                'model': model,
                'load_time': load_time
            }
            
            print(f"✅ {model_name} loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return None
    
    print(f"\n🔍 Comparing {len(models)} models...")
    print("=" * 60)
    
    # Transcribe with each model
    results = {
        'timestamp': datetime.now().isoformat(),
        'audio_duration': len(audio) / sample_rate,
        'sample_rate': sample_rate,
        'models': {}
    }
    
    for model_name, model_info in models.items():
        result = transcribe_with_model(audio, model_info['model'], model_name)
        
        if result:
            results['models'][model_name] = result
            
            print(f"\n📝 {model_name.upper()} Model Results:")
            print(f"   Transcription: '{result['transcription']}'")
            print(f"   Time: {result['transcription_time']:.2f}s")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Language: {result['language']}")
    
    return results

def print_comparison_summary(results):
    """Print a summary comparison of all models"""
    if not results or not results['models']:
        print("❌ No results to display")
        return
    
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
    
    # Accuracy comparison
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
    parser = argparse.ArgumentParser(description='Compare Whisper models on an audio file')
    parser.add_argument('audio_file', help='Path to the audio file to transcribe')
    parser.add_argument('--models', nargs='+', default=['tiny', 'base', 'small'], 
                       help='Models to compare (default: tiny base small)')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("🔬 Whisper Model Comparison Tool")
    print("=" * 50)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    # Load audio
    print(f"📁 Loading audio file: {args.audio_file}")
    audio, sample_rate = load_audio_file(args.audio_file)
    
    if audio is None:
        return
    
    print(f"✅ Audio loaded: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")
    
    # Compare models
    results = compare_models_on_audio(audio, sample_rate, args.models)
    
    if results:
        # Print summary
        print_comparison_summary(results)
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        else:
            # Auto-save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"whisper_comparison_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        print(f"\n🎉 Comparison complete!")

if __name__ == "__main__":
    main() 