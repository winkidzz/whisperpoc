#!/usr/bin/env python3
"""
Voice-to-Direct-LLM Multimodal System
Supports multiple providers: OpenAI, Anthropic, Google, and local models
"""

import sounddevice as sd
import numpy as np
import time
import json
import os
from datetime import datetime
import requests
import base64
from typing import Optional, Dict, Any

class VoiceToLLM:
    def __init__(self, provider="openai", api_key=None):
        """
        Initialize Voice-to-LLM system
        
        Args:
            provider: 'openai', 'anthropic', 'google', or 'local'
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.sample_rate = 16000
        self.conversation_history = []
        
        if not self.api_key and provider != "local":
            print(f"⚠️  Warning: No API key found for {provider}")
            print(f"   Set {provider.upper()}_API_KEY environment variable")
    
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
    
    def audio_to_base64(self, audio):
        """Convert audio to base64 for API transmission"""
        import io
        import scipy.io.wavfile as wav
        
        # Save to temporary buffer
        buffer = io.BytesIO()
        wav.write(buffer, self.sample_rate, audio)
        buffer.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    
    def call_openai_gpt4o(self, audio, prompt=""):
        """Call OpenAI GPT-4o with voice input"""
        if not self.api_key:
            return "❌ OpenAI API key not found"
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Convert audio to base64
            audio_base64 = self.audio_to_base64(audio)
            
            # Prepare messages
            messages = []
            if prompt:
                messages.append({"role": "user", "content": prompt})
            
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please respond to this voice input."
                    },
                    {
                        "type": "audio",
                        "audio": {
                            "data": audio_base64,
                            "mime_type": "audio/wav"
                        }
                    }
                ]
            })
            
            # Call API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ OpenAI API error: {e}"
    
    def call_anthropic_claude(self, audio, prompt=""):
        """Call Anthropic Claude with voice input"""
        if not self.api_key:
            return "❌ Anthropic API key not found"
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Convert audio to base64
            audio_base64 = self.audio_to_base64(audio)
            
            # Prepare content
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            
            content.append({
                "type": "audio",
                "source": {
                    "type": "base64",
                    "media_type": "audio/wav",
                    "data": audio_base64
                }
            })
            
            # Call API
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                content=content
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"❌ Anthropic API error: {e}"
    
    def call_google_gemini(self, audio, prompt=""):
        """Call Google Gemini with voice input"""
        if not self.api_key:
            return "❌ Google API key not found"
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Convert audio to base64
            audio_base64 = self.audio_to_base64(audio)
            
            # Prepare content
            content = []
            if prompt:
                content.append(prompt)
            
            content.append({
                "mime_type": "audio/wav",
                "data": audio_base64
            })
            
            # Call API
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(content)
            
            return response.text
            
        except Exception as e:
            return f"❌ Google API error: {e}"
    
    def call_local_model(self, audio, prompt=""):
        """Call local multimodal model (placeholder for future implementation)"""
        # This is a placeholder for local multimodal models
        # Currently, most local models don't support direct audio input
        # You would need to use Whisper + local LLM approach
        
        return "🔧 Local multimodal models with direct audio support are still in development.\n" \
               "For now, use the Whisper + local LLM approach we built earlier."
    
    def process_voice_input(self, audio, prompt="", save_conversation=True):
        """Process voice input with the selected provider"""
        print(f"\n🤖 Processing with {self.provider.upper()}...")
        
        # Call appropriate provider
        if self.provider == "openai":
            response = self.call_openai_gpt4o(audio, prompt)
        elif self.provider == "anthropic":
            response = self.call_anthropic_claude(audio, prompt)
        elif self.provider == "google":
            response = self.call_google_gemini(audio, prompt)
        elif self.provider == "local":
            response = self.call_local_model(audio, prompt)
        else:
            response = f"❌ Unknown provider: {self.provider}"
        
        # Save conversation
        if save_conversation:
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider,
                "prompt": prompt,
                "response": response
            }
            self.conversation_history.append(conversation_entry)
        
        return response
    
    def start_interactive_session(self, duration=10):
        """Start interactive voice-to-LLM session"""
        print(f"🎧 Voice-to-LLM Interactive Session")
        print(f"🤖 Provider: {self.provider.upper()}")
        print("=" * 50)
        print("💡 Speak your questions or requests")
        print("⏹️  Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Record audio
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = f"voice_input_{timestamp}.wav"
                
                audio = self.record_audio(duration, audio_file)
                
                # Process with LLM
                response = self.process_voice_input(audio)
                
                # Display response
                print(f"\n🤖 {self.provider.upper()} Response:")
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
            filename = f"conversation_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"💾 Conversation history saved to: {filename}")

def main():
    """Main function"""
    print("🎤 Voice-to-Direct-LLM Multimodal System")
    print("=" * 50)
    
    # Provider selection
    providers = {
        "1": "openai",
        "2": "anthropic", 
        "3": "google",
        "4": "local"
    }
    
    print("\nSelect provider:")
    for key, provider in providers.items():
        print(f"  {key}. {provider.upper()}")
    
    while True:
        choice = input("\nEnter your choice (1-4, default=1): ").strip()
        if not choice:
            choice = "1"
        
        if choice in providers:
            provider = providers[choice]
            break
        else:
            print("Invalid choice. Please select 1-4.")
    
    # Get API key if needed
    api_key = None
    if provider != "local":
        api_key = input(f"\nEnter {provider.upper()} API key (or press Enter to use environment variable): ").strip()
        if not api_key:
            api_key = None
    
    # Initialize system
    voice_llm = VoiceToLLM(provider=provider, api_key=api_key)
    
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
    
    # Start session
    voice_llm.start_interactive_session(duration)

if __name__ == "__main__":
    main() 