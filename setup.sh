#!/bin/bash

# Whisper Live Transcriber Setup Script
# This script automates the installation process for different operating systems

echo "🤖 Whisper Live Transcriber Setup"
echo "=================================="

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "📱 Detected OS: $OS"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "📦 Installing Python dependencies..."

# Install Python dependencies
if pip3 install -r requirements.txt; then
    echo "✅ Python dependencies installed successfully!"
else
    echo "❌ Failed to install Python dependencies"
    echo "💡 Try installing manually: pip3 install -r requirements.txt"
    exit 1
fi

# Install system dependencies based on OS
case $OS in
    "macos")
        echo "🍎 Installing macOS dependencies..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "📦 Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install PortAudio
        if brew install portaudio; then
            echo "✅ PortAudio installed successfully!"
        else
            echo "❌ Failed to install PortAudio"
            echo "💡 Try installing manually: brew install portaudio"
        fi
        ;;
        
    "linux")
        echo "🐧 Installing Linux dependencies..."
        
        # Detect Linux distribution
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            echo "📦 Installing dependencies for Debian/Ubuntu..."
            sudo apt-get update
            if sudo apt-get install -y portaudio19-dev python3-pyaudio; then
                echo "✅ Linux dependencies installed successfully!"
            else
                echo "❌ Failed to install Linux dependencies"
                echo "💡 Try installing manually: sudo apt-get install portaudio19-dev python3-pyaudio"
            fi
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            echo "📦 Installing dependencies for CentOS/RHEL..."
            sudo yum install -y portaudio-devel python3-pyaudio
        elif command -v pacman &> /dev/null; then
            # Arch Linux
            echo "📦 Installing dependencies for Arch Linux..."
            sudo pacman -S portaudio python-pyaudio
        else
            echo "⚠️  Unknown Linux distribution. Please install portaudio19-dev and python3-pyaudio manually."
        fi
        ;;
        
    "windows")
        echo "🪟 Windows detected..."
        echo "💡 For Windows, you may need to install PyAudio manually:"
        echo "   pip3 install pipwin"
        echo "   pipwin install pyaudio"
        ;;
esac

echo ""
echo "🎉 Setup completed!"
echo ""
echo "🚀 To start using Whisper Live Transcriber:"
echo "   python3 whisper_live.py    # Advanced mode (recommended)"
echo "   python3 simple_whisper.py  # Simple mode"
echo ""
echo "📖 For more information, see README.md" 