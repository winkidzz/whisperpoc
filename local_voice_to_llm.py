#!/usr/bin/env python3
"""
Local Voice-to-LLM Multimodal System
Combines Whisper with local multimodal LLMs for voice processing
"""

import whisper
import sounddevice as sd
import numpy as np
import time
import json
import os
import subprocess
import requests
from datetime import datetime
from typing import Optional, Dict, Any

class LocalVoiceToLLM:
    def __init__(self, whisper_model="base", llm_provider="ollama", llm_model="llava"):
        """
        Initialize local voice-to-LLM system
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            llm_provider: 'ollama', 'localai', or 'custom'
            llm_model: Model name for the LLM provider
        """
        self.whisper_model = whisper_model
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.sample_rate = 16000
        self.conversation_history = []
        
        # Initialize Whisper
        print(f"🤖 Loading Whisper model: {whisper_model}")
        self.whisper = whisper.load_model(whisper_model)
        print("✅ Whisper loaded successfully!")
        
        # Check LLM provider
        self.check_llm_provider()
    
    def check_llm_provider(self):
        """Check if the LLM provider is available"""
        if self.llm_provider == "ollama":
            try:
                # Check if Ollama is running
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [model["name"] for model in models]
                    print(f"✅ Ollama available. Models: {available_models}")
                    
                    if self.llm_model not in available_models:
                        print(f"⚠️  Model {self.llm_model} not found. Available: {available_models}")
                        print(f"💡 You can pull it with: ollama pull {self.llm_model}")
                else:
                    print("❌ Ollama not responding properly")
            except requests.exceptions.RequestException:
                print("❌ Ollama not running. Start with: ollama serve")
                print("💡 Install Ollama: https://ollama.ai")
        
        elif self.llm_provider == "localai":
            try:
                response = requests.get("http://localhost:8080/v1/models", timeout=5)
                if response.status_code == 200:
                    print("✅ LocalAI available")
                else:
                    print("❌ LocalAI not responding properly")
            except requests.exceptions.RequestException:
                print("❌ LocalAI not running")
                print("💡 Install LocalAI: https://github.com/go-skynet/LocalAI")
    
    def record_audio(self, duration=10, save_file=None):
        """Record audio from microphone"""
        print(f"🎤 Recording {duration} seconds...")
        print("💡 Speak your question or request")
        
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
        
        if save_file:
            import scipy.io.wavfile as wav
            wav.write(save_file, self.sample_rate, audio)
            print(f"💾 Audio saved to: {save_file}")
        
        return audio
    
    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        print("🎯 Transcribing audio with Whisper...")
        
        try:
            start_time = time.time()
            
            result = self.whisper.transcribe(
                audio,
                language="en",
                task="transcribe",
                fp16=False
            )
            
            transcription_time = time.time() - start_time
            transcription = result["text"].strip()
            
            print(f"✅ Transcription ({transcription_time:.2f}s): '{transcription}'")
            
            return {
                'transcription': transcription,
                'transcription_time': transcription_time,
                'confidence': result.get('confidence', 0.0),
                'language': result.get('language', 'en')
            }
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def call_ollama_llm(self, text, image_path=None):
        """Call Ollama LLM with text and optional image"""
        try:
            # Prepare the request
            request_data = {
                "model": self.llm_model,
                "prompt": text,
                "stream": False
            }
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                import base64
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                request_data["images"] = [image_data]
            
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response")
            else:
                return f"❌ Ollama API error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Ollama error: {e}"
    
    def call_localai_llm(self, text, image_path=None):
        """Call LocalAI LLM with text and optional image"""
        try:
            # Prepare the request
            request_data = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "stream": False
            }
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                import base64
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                request_data["messages"][0]["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            
            # Call LocalAI API
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"❌ LocalAI API error: {response.status_code}"
                
        except Exception as e:
            return f"❌ LocalAI error: {e}"
    
    def call_custom_llm(self, text, image_path=None):
        """Call custom local LLM (placeholder for custom implementations)"""
        # This is a placeholder for custom local LLM implementations
        # You can integrate with your own local model here
        
        return f"🔧 Custom LLM placeholder\n" \
               f"Text: {text}\n" \
               f"Image: {image_path if image_path else 'None'}\n" \
               f"Implement your local LLM integration here."
    
    def process_voice_input(self, audio, image_path=None, save_conversation=True):
        """Process voice input with local LLM"""
        print(f"\n🤖 Processing with local {self.llm_provider.upper()}...")
        
        # Step 1: Transcribe audio
        transcription_result = self.transcribe_audio(audio)
        
        if not transcription_result:
            return "❌ Transcription failed"
        
        transcription = transcription_result['transcription']
        
        if not transcription.strip():
            return "🔇 No speech detected"
        
        # Step 2: Send to local LLM
        if self.llm_provider == "ollama":
            response = self.call_ollama_llm(transcription, image_path)
        elif self.llm_provider == "localai":
            response = self.call_localai_llm(transcription, image_path)
        elif self.llm_provider == "custom":
            response = self.call_custom_llm(transcription, image_path)
        else:
            response = f"❌ Unknown LLM provider: {self.llm_provider}"
        
        # Save conversation
        if save_conversation:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "whisper_model": self.whisper_model,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "transcription": transcription,
                "transcription_time": transcription_result['transcription_time'],
                "image_path": image_path,
                "response": response
            }
            self.conversation_history.append(conversation_entry)
        
        return response
    
    def start_interactive_session(self, duration=10):
        """Start interactive local voice-to-LLM session"""
        print(f"🎧 Local Voice-to-LLM Interactive Session")
        print(f"🤖 Whisper: {self.whisper_model}")
        print(f"🤖 LLM: {self.llm_provider} ({self.llm_model})")
        print("=" * 50)
        print("💡 Speak your questions or requests")
        print("🖼️  You can also provide image files for multimodal processing")
        print("⏹️  Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Record audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = f"local_voice_input_{timestamp}.wav"
                
                audio = self.record_audio(duration, audio_file)
                
                # Ask for image (optional)
                image_path = input("\n🖼️  Enter image path (or press Enter to skip): ").strip()
                if not image_path or not os.path.exists(image_path):
                    image_path = None
                    if image_path:
                        print("⚠️  Image file not found, proceeding without image")
                
                # Process with local LLM
                response = self.process_voice_input(audio, image_path)
                
                # Display response
                print(f"\n🤖 {self.llm_provider.upper()} Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                print()
                
        except KeyboardInterrupt:
            print("\n⏹️  Session ended")
            self.save_conversation_history()
    
    def save_conversation_history(self, filename=None):
        """Save conversation history to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"local_conversation_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"💾 Conversation history saved to: {filename}")

def main():
    """Main function"""
    print("🏠 Local Voice-to-LLM Multimodal System")
    print("=" * 50)
    
    # Whisper model selection
    whisper_models = {
        "1": "tiny",
        "2": "base", 
        "3": "small",
        "4": "medium",
        "5": "large"
    }
    
    print("\nSelect Whisper model:")
    for key, model in whisper_models.items():
        print(f"  {key}. {model}")
    
    while True:
        choice = input("\nEnter your choice (1-5, default=2): ").strip()
        if not choice:
            choice = "2"
        
        if choice in whisper_models:
            whisper_model = whisper_models[choice]
            break
        else:
            print("Invalid choice. Please select 1-5.")
    
    # LLM provider selection
    llm_providers = {
        "1": ("ollama", "llava"),
        "2": ("ollama", "llava:7b"),
        "3": ("ollama", "llava:13b"),
        "4": ("localai", "llava"),
        "5": ("custom", "custom")
    }
    
    print("\nSelect LLM provider:")
    print("  1. Ollama (llava)")
    print("  2. Ollama (llava:7b)")
    print("  3. Ollama (llava:13b)")
    print("  4. LocalAI (llava)")
    print("  5. Custom")
    
    while True:
        choice = input("\nEnter your choice (1-5, default=1): ").strip()
        if not choice:
            choice = "1"
        
        if choice in llm_providers:
            llm_provider, llm_model = llm_providers[choice]
            break
        else:
            print("Invalid choice. Please select 1-5.")
    
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
    
    # Initialize system
    voice_llm = LocalVoiceToLLM(
        whisper_model=whisper_model,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    # Start session
    voice_llm.start_interactive_session(duration)

if __name__ == "__main__":
    main() 