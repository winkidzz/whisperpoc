# WhisperPOC Project Setup Guide

## 🏗️ **Project Overview**

WhisperPOC is a comprehensive audio recording, transcription, and storage system that demonstrates:
- Real-time audio recording with OpenAI Whisper
- PostgreSQL database storage with vector embeddings
- Interactive recording sessions with metadata
- Vector similarity search and analysis
- Voice-to-LLM integration examples

## 📁 **Project Structure**

```
whisperpoc/
├── 📄 Documentation
│   ├── README.md                    # Main project overview
│   ├── PROJECT_SETUP.md            # This file - complete setup guide
│   ├── DATABASE_SETUP.md           # Database initialization guide
│   ├── API_REFERENCE.md            # Complete API documentation
│   ├── EXAMPLES.md                 # Usage examples and tutorials
│   └── TROUBLESHOOTING.md          # Common issues and solutions
│
├── 🐍 Core Applications
│   ├── audio_storage_system.py     # Main storage system
│   ├── interactive_recorder.py     # Interactive recording interface
│   ├── embedding_queries.py        # Vector embedding query engine
│   └── postgres_demo.py           # PostgreSQL demonstration
│
├── 🎤 Audio Processing
│   ├── whisper_live.py            # Real-time Whisper transcription
│   ├── simple_whisper.py          # Simple Whisper implementation
│   ├── test_audio.py              # Audio testing utilities
│   ├── model_comparison.py        # Whisper model comparison
│   └── compare_audio_file.py      # Audio file comparison
│
├── 🤖 Voice-to-LLM Integration
│   ├── voice_to_llm.py            # Cloud API voice-to-LLM
│   ├── local_voice_to_llm.py      # Local voice-to-LLM system
│   └── requirements_voice_llm.txt # Voice-to-LLM dependencies
│
├── 📊 Test Data & Examples
│   ├── test_data/                 # Sample audio files and data
│   ├── examples/                  # Example scripts and notebooks
│   └── sample_queries.sql         # Sample database queries
│
├── ⚙️ Configuration
│   ├── requirements.txt           # Core dependencies
│   ├── setup.sh                  # Automated setup script
│   ├── .gitignore               # Git ignore rules
│   └── config.py                # Configuration settings
│
└── 📚 Guides & Comparisons
    ├── AUDIO_STORAGE_GUIDE.md    # Storage system guide
    ├── EMBEDDING_QUERY_GUIDE.md  # Vector embedding guide
    ├── POSTGRES_SUMMARY.md       # PostgreSQL implementation
    ├── LOCAL_VOICE_TO_LLM_GUIDE.md # Local voice-to-LLM guide
    └── VOICE_TO_LLM_COMPARISON.md # System comparison
```

## 🚀 **Quick Start (5 Minutes)**

### **1. Prerequisites Check**
```bash
# Check Python version (3.8+ required)
python3 --version

# Check if PostgreSQL is installed
psql --version

# Check if Homebrew is available (macOS)
brew --version
```

### **2. Automated Setup**
```bash
# Clone and setup
git clone https://github.com/winkidzz/whisperpoc.git
cd whisperpoc
chmod +x setup.sh
./setup.sh
```

### **3. Database Initialization**
```bash
# Create database
createdb audio_storage_db

# Initialize tables (automatic on first run)
python audio_storage_system.py --init-only
```

### **4. Test the System**
```bash
# Test audio recording
python test_audio.py

# Run interactive recorder
python interactive_recorder.py

# Test embedding queries
python embedding_queries.py
```

## 🗄️ **Database Setup**

### **PostgreSQL Installation**

#### **macOS (Homebrew)**
```bash
brew install postgresql
brew services start postgresql
```

#### **Ubuntu/Debian**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### **Windows**
Download from: https://www.postgresql.org/download/windows/

### **Database Creation**
```bash
# Create database
createdb audio_storage_db

# Or using psql
psql -U postgres
CREATE DATABASE audio_storage_db;
\q
```

### **Tables Structure**
The system automatically creates these tables on first run:

```sql
-- Audio files metadata
CREATE TABLE audio_files (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Whisper transcriptions
CREATE TABLE transcripts (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    transcription TEXT NOT NULL,
    confidence REAL,
    language VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL,
    embedding_data BYTEA NOT NULL,
    embedding_dimension INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Flexible metadata
CREATE TABLE metadata (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 **Application Quick Reference**

### **Core Applications**

#### **1. Interactive Recorder**
```bash
python interactive_recorder.py
```
- **Purpose**: Record audio with custom metadata
- **Features**: Duration selection, Whisper model choice, metadata input
- **Output**: Stored in PostgreSQL with embeddings

#### **2. Audio Storage System**
```bash
python audio_storage_system.py
```
- **Purpose**: Programmatic audio recording and storage
- **Features**: Batch recording, automatic transcription, embedding generation
- **API**: Full Python API for integration

#### **3. Embedding Query Engine**
```bash
python embedding_queries.py
```
- **Purpose**: Query and analyze vector embeddings
- **Features**: Similarity search, metadata filtering, export capabilities
- **Output**: Similar audio files, statistical analysis

#### **4. PostgreSQL Demo**
```bash
python postgres_demo.py
```
- **Purpose**: Demonstrate database features
- **Features**: Batch operations, performance monitoring, data export
- **Output**: Database statistics and sample queries

### **Audio Processing Tools**

#### **5. Real-time Whisper**
```bash
python whisper_live.py
```
- **Purpose**: Continuous real-time transcription
- **Features**: Voice Activity Detection, live streaming
- **Use Case**: Live speech-to-text

#### **6. Model Comparison**
```bash
python model_comparison.py
```
- **Purpose**: Compare Whisper model performance
- **Features**: Accuracy, speed, confidence comparison
- **Output**: JSON comparison results

### **Voice-to-LLM Integration**

#### **7. Cloud Voice-to-LLM**
```bash
python voice_to_llm.py
```
- **Purpose**: Voice interaction with cloud LLMs
- **APIs**: OpenAI GPT-4o, Anthropic Claude, Google Gemini
- **Features**: Real-time conversation

#### **8. Local Voice-to-LLM**
```bash
python local_voice_to_llm.py
```
- **Purpose**: Local voice-to-LLM system
- **Backends**: Ollama, LocalAI
- **Features**: Privacy-focused, offline operation

## 🔧 **Configuration**

### **Environment Variables**
Create a `.env` file for sensitive configuration:

```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=audio_storage_db
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password

# API Keys (for voice-to-LLM)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
```

### **Configuration File**
The system uses `config.py` for default settings:

```python
# Default PostgreSQL configuration
DEFAULT_POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'audio_storage_db',
    'user': 'postgres',
    'password': ''
}

# Default Whisper settings
DEFAULT_WHISPER_MODEL = 'base'
DEFAULT_RECORDING_DURATION = 10
DEFAULT_SAMPLE_RATE = 16000
```

## 📊 **Test Data & Examples**

### **Sample Audio Files**
The project includes test audio files in `test_data/`:
- `sample_meeting.wav` - 30-second meeting recording
- `sample_interview.wav` - 60-second interview
- `sample_presentation.wav` - 120-second presentation

### **Example Queries**
Sample SQL queries in `sample_queries.sql`:
```sql
-- Find all recordings by user
SELECT af.file_path, t.transcription, m.value as user
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'user' AND m.value = 'john';

-- Get recent recordings
SELECT * FROM audio_files 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Find high-confidence transcriptions
SELECT af.file_path, t.transcription, t.confidence
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
WHERE t.confidence > 0.9
ORDER BY t.confidence DESC;
```

### **Example Python Scripts**
Example scripts in `examples/`:
- `batch_recording.py` - Record multiple audio files
- `similarity_search.py` - Find similar audio content
- `export_data.py` - Export data to various formats
- `performance_analysis.py` - Analyze system performance

## 🧪 **Testing & Validation**

### **System Tests**
```bash
# Test audio recording
python test_audio.py

# Test database connection
python -c "from audio_storage_system import AudioStorageSystem; AudioStorageSystem(storage_backend='postgres')"

# Test embedding generation
python -c "from embedding_queries import EmbeddingQueryEngine; EmbeddingQueryEngine()"
```

### **Performance Benchmarks**
```bash
# Run performance tests
python examples/performance_analysis.py

# Test different Whisper models
python model_comparison.py --duration 30 --models tiny,base,small
```

### **Data Validation**
```bash
# Validate database integrity
python -c "
from audio_storage_system import AudioStorageSystem
storage = AudioStorageSystem(storage_backend='postgres')
print(f'Audio files: {storage.get_audio_count()}')
print(f'Transcripts: {storage.get_transcript_count()}')
print(f'Embeddings: {storage.get_embedding_count()}')
"
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Audio Device Not Found**
```bash
# Check available audio devices
python test_audio.py --list-devices

# Install audio dependencies
brew install portaudio  # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

#### **2. PostgreSQL Connection Failed**
```bash
# Check PostgreSQL status
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql      # Ubuntu

# Test connection
psql -d audio_storage_db -c "SELECT version();"
```

#### **3. Whisper Model Download Failed**
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper
python -c "import whisper; whisper.load_model('base')"
```

#### **4. Memory Issues**
```bash
# Use smaller Whisper model
export WHISPER_MODEL=tiny

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

### **Debug Mode**
Enable debug logging:
```bash
export DEBUG=1
python audio_storage_system.py --debug
```

## 📈 **Performance Optimization**

### **Database Optimization**
```sql
-- Create indexes for better performance
CREATE INDEX idx_audio_files_created_at ON audio_files(created_at);
CREATE INDEX idx_transcripts_confidence ON transcripts(confidence);
CREATE INDEX idx_metadata_key_value ON metadata(key, value);

-- Analyze tables for query optimization
ANALYZE audio_files;
ANALYZE transcripts;
ANALYZE embeddings;
ANALYZE metadata;
```

### **Whisper Optimization**
```python
# Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use smaller models for faster processing
model = whisper.load_model("tiny")  # Fastest
model = whisper.load_model("base")  # Balanced
```

### **Memory Management**
```python
# Clear GPU memory after processing
import torch
torch.cuda.empty_cache()

# Use generators for large datasets
def audio_file_generator():
    for file in audio_files:
        yield process_audio(file)
```

## 🔮 **Next Steps**

### **Immediate Enhancements**
1. **Real Embeddings**: Implement actual embedding generation
2. **Vector Database**: Add pgvector for native vector operations
3. **Web Interface**: Create browser-based recording interface
4. **API Endpoints**: Add RESTful API for remote access

### **Advanced Features**
1. **Multi-modal Embeddings**: Audio + text combined embeddings
2. **Real-time Processing**: Streaming audio transcription
3. **Cloud Storage**: S3/Google Cloud integration
4. **Advanced Search**: Semantic search with multiple modalities

### **Production Deployment**
1. **Docker Containerization**: Containerize the application
2. **Environment Management**: Proper environment configuration
3. **Monitoring**: Add logging and monitoring
4. **Security**: Implement authentication and authorization

---

**This guide ensures that any developer (including LLM editors like Cursor) can quickly understand, set up, and run the WhisperPOC project with minimal friction.** 