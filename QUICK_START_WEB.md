# 🚀 Quick Start Guide - WhisperPOC Web Interface

## Prerequisites
- Python 3.8+
- PostgreSQL database (configured in `config.py`)
- Microphone access

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Database Setup
Make sure your PostgreSQL database is running and configured in `config.py`:
```bash
# Check database connection
python postgres_demo.py
```

### 3. Start the Web Interface

**Option A: Using the helper script (Recommended)**
```bash
python start_web_interface.py
```

**Option B: Manual startup**
```bash
cd web_interface
python app.py
```

### 4. Access the Web Interface
Open your browser and go to: **http://localhost:5002**

## 🎯 Available Features

### 📊 Dashboard (`/`)
- View statistics (audio files, transcripts, embeddings)
- Quick actions for recording and searching
- Recent recordings overview

### 🎤 Record Audio (`/record`)
- Real-time audio recording
- Live transcription with Whisper
- Configurable recording duration
- Automatic embedding generation

### 📁 Browse Files (`/browse`)
- View all audio files with pagination
- Filter by confidence and language
- Download, play, or delete files
- View detailed metadata

### 🔍 Search Content (`/search`)
- Semantic search through transcripts
- Configurable similarity threshold
- Real-time search results

## 🔧 API Endpoints

### Health Check
```bash
curl http://localhost:5002/health
```

### Get Statistics
```bash
curl http://localhost:5002/api/statistics
```

### Record Audio
```bash
curl -X POST http://localhost:5002/api/record \
  -H "Content-Type: application/json" \
  -d '{"duration": 10, "whisper_model": "base"}'
```

## 🛠️ Troubleshooting

### Port Already in Use
If port 5002 is busy:
```bash
# Kill existing process
lsof -ti:5002 | xargs kill -9

# Or use a different port
cd web_interface
python app.py --port 5003
```

### Database Connection Issues
```bash
# Check database status
python postgres_demo.py

# Verify configuration
python config.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📱 Web Interface Features

### Real-time Recording
- Click "Start Recording" to begin
- Watch live transcription as you speak
- Automatic file saving and embedding generation

### File Management
- Browse all recorded audio files
- Filter and search through content
- Download or delete files as needed

### Semantic Search
- Search for specific topics or phrases
- Adjust similarity threshold for better results
- View matching audio files with timestamps

## 🔒 Security Notes
- This is a development server - not for production use
- Debug mode is enabled for development
- Consider using a production WSGI server for deployment

## 📞 Support
- Check `WEB_INTERFACE_README.md` for detailed documentation
- Review `TROUBLESHOOTING.md` for common issues
- Check the logs in the terminal for error messages

---

**🎉 You're all set! The web interface is now running and ready to use.** 