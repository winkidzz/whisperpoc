# WhisperPOC Web Interface

A modern, responsive web interface for the WhisperPOC audio storage and transcription system. This interface provides an intuitive way to record, manage, and search through your audio files with real-time transcription capabilities.

## Features

### 🎤 Audio Recording
- **Real-time Recording**: Record audio directly through your browser
- **Multiple Whisper Models**: Choose from tiny to large models for speed vs accuracy
- **Customizable Duration**: Set recording duration from 5 to 300 seconds
- **Metadata Support**: Add custom metadata to your recordings
- **Progress Tracking**: Visual feedback during recording

### 📁 File Management
- **Browse Audio Files**: View all your recordings with pagination
- **Search & Filter**: Filter by confidence, language, and search transcriptions
- **Bulk Operations**: Select and delete multiple files
- **File Details**: View comprehensive information about each recording
- **Audio Playback**: Play audio files directly in the browser

### 🔍 Semantic Search
- **Natural Language Queries**: Search using everyday language
- **Semantic Similarity**: Find related content using embeddings
- **Configurable Thresholds**: Adjust similarity sensitivity
- **Search History**: Keep track of your recent searches
- **Multiple Result Limits**: Get 5 to 50 results per search

### 📊 Dashboard
- **System Statistics**: View audio count, transcript count, and embedding count
- **Recent Activity**: See your latest recordings
- **Quick Actions**: Easy access to common functions
- **Export Capabilities**: Download data in JSON or CSV format

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Web Interface

```bash
python start_web_interface.py
```

Or directly:

```bash
cd web_interface
python app.py
```

### 3. Access the Interface

Open your browser and navigate to: **http://localhost:5002**

## System Requirements

### Software Dependencies
- Python 3.8+
- PostgreSQL database
- Whisper models (downloaded automatically)

### Python Packages
- Flask (web framework)
- Flask-SocketIO (real-time features)
- psycopg2-binary (PostgreSQL adapter)
- openai-whisper (transcription)
- torch (machine learning)
- sounddevice (audio recording)
- numpy, scipy, librosa (audio processing)

### Hardware Requirements
- Microphone for audio recording
- Sufficient storage for audio files
- RAM: 4GB+ (8GB+ recommended for larger models)

## Configuration

### Database Setup
The web interface uses the same PostgreSQL configuration as the main WhisperPOC system. Make sure your database is properly configured in `config.py`.

### Audio Devices
The interface will automatically detect available audio input devices. If you have multiple microphones, the system will use the default device.

### Whisper Models
Available models and their characteristics:

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| Tiny | Very Fast | Basic | ~39MB | Real-time applications |
| Base | Fast | Good | ~74MB | General use (default) |
| Small | Medium | Better | ~244MB | Better accuracy needed |
| Medium | Slow | High | ~769MB | Important recordings |
| Large | Very Slow | Best | ~1550MB | Maximum accuracy |

## Usage Guide

### Recording Audio

1. **Navigate to Record Page**: Click "Record" in the navigation
2. **Configure Settings**:
   - Set recording duration (5-300 seconds)
   - Choose Whisper model
   - Add optional metadata (JSON format)
3. **Start Recording**: Click "Start Recording"
4. **Monitor Progress**: Watch the progress bar and time elapsed
5. **Review Results**: View transcription and confidence score

### Browsing Files

1. **Access Browse Page**: Click "Browse" in the navigation
2. **Use Filters**:
   - Search transcriptions by text
   - Filter by confidence level
   - Filter by language
3. **View Details**: Click the eye icon to see full information
4. **Play Audio**: Click the play icon to listen
5. **Download Files**: Click the download icon
6. **Delete Files**: Use the trash icon (with confirmation)

### Searching Content

1. **Go to Search Page**: Click "Search" in the navigation
2. **Enter Query**: Use natural language (e.g., "meeting about budget")
3. **Adjust Settings**:
   - Set number of results (5-50)
   - Choose similarity threshold
4. **View Results**: See similarity scores and transcriptions
5. **Access Files**: Click through to view, play, or download

### Dashboard Overview

The dashboard provides:
- **Statistics Cards**: Audio count, transcript count, embeddings, today's recordings
- **Quick Actions**: Direct links to record, browse, search, and export
- **Recent Recordings**: Latest audio files with basic info
- **Export Options**: Download data in various formats

## API Endpoints

The web interface provides several REST API endpoints:

### Audio Management
- `GET /api/audio/<id>` - Get audio file information
- `GET /api/audio/<id>/download` - Download audio file
- `DELETE /api/audio/<id>/delete` - Delete audio file
- `POST /api/record` - Record new audio

### Search
- `POST /api/search` - Perform semantic search

### Statistics
- `GET /api/statistics` - Get system statistics
- `GET /api/export` - Export data (JSON/CSV)

### Health Check
- `GET /health` - System health status

## Troubleshooting

### Common Issues

**Database Connection Failed**
- Ensure PostgreSQL is running
- Check database credentials in `config.py`
- Verify database exists and is accessible

**Audio Recording Not Working**
- Check microphone permissions in browser
- Ensure audio device is properly connected
- Try refreshing the page

**Whisper Model Download Issues**
- Check internet connection
- Ensure sufficient disk space
- Models are downloaded automatically on first use

**Search Not Finding Results**
- Verify embeddings are generated for audio files
- Try lowering the similarity threshold
- Check if audio files have transcriptions

### Error Messages

**"No audio input devices found"**
- Connect a microphone
- Check system audio settings
- Restart the application

**"Database connection failed"**
- Start PostgreSQL service
- Verify connection parameters
- Check firewall settings

**"Whisper model not found"**
- Wait for automatic download
- Check disk space
- Restart the application

## Development

### Project Structure
```
web_interface/
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── base.html       # Base template
│   ├── dashboard.html  # Dashboard page
│   ├── record.html     # Recording page
│   ├── browse.html     # Browse page
│   ├── search.html     # Search page
│   └── error.html      # Error page
└── static/             # Static assets (CSS, JS, images)
```

### Adding New Features
1. Add routes to `app.py`
2. Create templates in `templates/`
3. Add JavaScript functionality
4. Update navigation in `base.html`

### Customization
- Modify CSS in `base.html` for styling
- Add new API endpoints in `app.py`
- Extend templates for additional functionality

## Security Considerations

- The web interface runs on localhost by default
- No authentication is implemented (add if needed for production)
- Database credentials should be kept secure
- Audio files are stored locally

## Performance Tips

- Use smaller Whisper models for faster processing
- Limit search results for better performance
- Regularly clean up old audio files
- Monitor disk space usage
- Use SSD storage for better I/O performance

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the main WhisperPOC documentation
3. Check system logs for error details
4. Ensure all dependencies are properly installed

## License

This web interface is part of the WhisperPOC project and follows the same licensing terms. 