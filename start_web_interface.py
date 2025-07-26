#!/usr/bin/env python3
"""
WhisperPOC Web Interface Startup Script

This script starts the web interface with proper error handling and setup.
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'flask',
        'flask_socketio',
        'psycopg2',
        'openai_whisper',
        'torch',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_database():
    """Check if database is accessible."""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from config import get_postgres_config
        
        config = get_postgres_config()
        import psycopg2
        
        conn = psycopg2.connect(**config)
        conn.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Make sure PostgreSQL is running and configured properly")
        return False

def check_audio_devices():
    """Check if audio devices are available."""
    try:
        import sounddevice
        devices = sounddevice.query_devices()
        input_devices = [d for d in devices if d['max_inputs'] > 0]
        
        if input_devices:
            print(f"✅ Found {len(input_devices)} audio input device(s)")
            return True
        else:
            print("⚠️  No audio input devices found")
            return False
    except Exception as e:
        print(f"⚠️  Could not check audio devices: {e}")
        return False

def main():
    """Main startup function."""
    print("=" * 60)
    print("WhisperPOC Web Interface Startup")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check database
    print("\n🔍 Checking database connection...")
    if not check_database():
        print("\n💡 You can still start the interface, but recording may not work")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check audio devices
    print("\n🔍 Checking audio devices...")
    check_audio_devices()
    
    # Start the web interface
    print("\n🚀 Starting web interface...")
    print("📱 The interface will be available at: http://localhost:5002")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Change to web_interface directory and start Flask app
        os.chdir('web_interface')
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Web interface stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 