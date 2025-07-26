#!/usr/bin/env python3
"""
Audio Test Script
Test microphone input and audio processing before running Whisper
"""

import sounddevice as sd
import numpy as np
import time

def test_microphone():
    """Test microphone input"""
    print("🎤 Testing microphone input...")
    print("💡 Speak into your microphone for 3 seconds")
    
    # Record 3 seconds of audio
    duration = 3
    sample_rate = 16000
    
    print(f"📹 Recording {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    
    # Show recording progress
    for i in range(duration):
        print(f"   Recording... {i+1}/{duration}")
        time.sleep(1)
    
    sd.wait()
    
    # Analyze audio
    audio = audio.flatten()
    max_amplitude = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    print(f"\n📊 Audio Analysis:")
    print(f"   Max amplitude: {max_amplitude:.4f}")
    print(f"   RMS level: {rms:.4f}")
    
    if max_amplitude > 0.01:
        print("✅ Microphone is working!")
        return True
    else:
        print("❌ No audio detected. Check your microphone.")
        return False

def list_audio_devices():
    """List available audio devices"""
    print("🔊 Available Audio Devices:")
    print("=" * 50)
    
    try:
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            device_type = []
            
            # Handle different property names
            max_inputs = device.get('max_inputs', device.get('maxInputs', 0))
            max_outputs = device.get('max_outputs', device.get('maxOutputs', 0))
            default_samplerate = device.get('default_samplerate', device.get('defaultSampleRate', 'Unknown'))
            
            if max_inputs > 0:
                device_type.append("input")
            if max_outputs > 0:
                device_type.append("output")
            
            print(f"  {i}: {device['name']} ({', '.join(device_type)})")
            print(f"      Sample rates: {default_samplerate} Hz")
            print(f"      Input channels: {max_inputs}")
            print(f"      Output channels: {max_outputs}")
            print()
            
    except Exception as e:
        print(f"⚠️  Could not list audio devices: {e}")
        print("   This is not critical for basic functionality.")

def test_whisper_import():
    """Test if Whisper can be imported"""
    try:
        import whisper
        print("✅ Whisper import successful!")
        return True
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        print("💡 Install Whisper: pip install openai-whisper")
        return False

def main():
    """Main test function"""
    print("🧪 Whisper Audio Test")
    print("=" * 30)
    
    # Test Whisper import
    print("\n1. Testing Whisper import...")
    whisper_ok = test_whisper_import()
    
    # List audio devices
    print("\n2. Listing audio devices...")
    list_audio_devices()
    
    # Test microphone
    print("\n3. Testing microphone...")
    mic_ok = test_microphone()
    
    # Summary
    print("\n" + "=" * 30)
    print("📋 Test Summary:")
    print(f"   Whisper: {'✅ OK' if whisper_ok else '❌ FAILED'}")
    print(f"   Microphone: {'✅ OK' if mic_ok else '❌ FAILED'}")
    
    if whisper_ok and mic_ok:
        print("\n🎉 All tests passed! You can now run:")
        print("   python3 whisper_live.py")
        print("   python3 simple_whisper.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        if not whisper_ok:
            print("💡 Install Whisper: pip install openai-whisper")
        if not mic_ok:
            print("💡 Check microphone permissions and connections")

if __name__ == "__main__":
    main() 