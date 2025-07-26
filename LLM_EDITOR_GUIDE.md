# WhisperPOC LLM Editor Guide

## 🎯 **For LLM Editors (Cursor, Copilot, etc.)**

This guide is specifically designed to help LLM editors like Cursor understand and work effectively with the WhisperPOC project.

## 📁 **Project Structure Overview**

```
whisperpoc/
├── 📄 Core Documentation
│   ├── README.md                    # Main project overview
│   ├── PROJECT_SETUP.md            # Complete setup guide
│   ├── DATABASE_SETUP.md           # PostgreSQL setup
│   ├── API_REFERENCE.md            # Full API documentation
│   ├── EXAMPLES.md                 # Usage examples
│   └── TROUBLESHOOTING.md          # Common issues
│
├── 🐍 Core Applications
│   ├── audio_storage_system.py     # Main storage system
│   ├── interactive_recorder.py     # Interactive recording
│   ├── embedding_queries.py        # Vector query engine
│   └── postgres_demo.py           # Database demo
│
├── 🎤 Audio Processing
│   ├── whisper_live.py            # Real-time transcription
│   ├── simple_whisper.py          # Simple transcription
│   ├── test_audio.py              # Audio testing
│   ├── model_comparison.py        # Model comparison
│   └── compare_audio_file.py      # Audio comparison
│
├── 🤖 Voice-to-LLM Integration
│   ├── voice_to_llm.py            # Cloud APIs
│   ├── local_voice_to_llm.py      # Local LLMs
│   └── requirements_voice_llm.txt # Dependencies
│
├── 📊 Examples & Test Data
│   ├── examples/                  # Example scripts
│   ├── test_data/                 # Sample data
│   └── sample_queries.sql         # SQL examples
│
├── ⚙️ Configuration
│   ├── config.py                  # Central configuration
│   ├── requirements.txt           # Dependencies
│   ├── setup.sh                  # Setup script
│   └── .gitignore               # Git ignore rules
│
└── 📚 Guides & Comparisons
    ├── AUDIO_STORAGE_GUIDE.md    # Storage guide
    ├── EMBEDDING_QUERY_GUIDE.md  # Embedding guide
    ├── POSTGRES_SUMMARY.md       # PostgreSQL summary
    ├── LOCAL_VOICE_TO_LLM_GUIDE.md # Local LLM guide
    └── VOICE_TO_LLM_COMPARISON.md # System comparison
```

## 🚀 **Quick Start for LLM Editors**

### **1. Understanding the Project**
This is a comprehensive audio recording, transcription, and storage system that:
- Records audio using Whisper for transcription
- Stores data in PostgreSQL with vector embeddings
- Provides similarity search and analysis
- Integrates with voice-to-LLM systems

### **2. Key Components to Focus On**
- **`audio_storage_system.py`**: Main system class
- **`embedding_queries.py`**: Vector similarity search
- **`config.py`**: Configuration management
- **`interactive_recorder.py`**: User interface

### **3. Database Schema**
```sql
-- Core tables
audio_files (id, file_hash, file_path, duration, etc.)
transcripts (id, audio_file_id, transcription, confidence)
embeddings (id, audio_file_id, embedding_data, type)
metadata (id, audio_file_id, key, value)
```

## 🔧 **Common Tasks for LLM Editors**

### **Task 1: Initialize Database**
```python
# The system automatically creates tables on first run
from audio_storage_system import AudioStorageSystem
storage = AudioStorageSystem(storage_backend="postgres")
```

### **Task 2: Record Audio**
```python
# Record and store audio with transcription
audio_id = storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True,
    metadata={"user": "john", "session": "meeting"}
)
```

### **Task 3: Search Similar Content**
```python
from embedding_queries import EmbeddingQueryEngine
query_engine = EmbeddingQueryEngine()

results = query_engine.search_by_text_similarity(
    query_text="meeting discussion",
    limit=5
)
```

### **Task 4: Export Data**
```python
# Export embeddings
export_file = query_engine.export_embeddings_to_numpy("embeddings.npz")

# Export to CSV
storage.cursor.execute("SELECT * FROM audio_files")
# ... CSV export logic
```

## 📋 **Configuration Management**

### **Environment Variables**
```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=audio_storage_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

# Whisper
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Embeddings
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536
```

### **Configuration Functions**
```python
from config import (
    get_postgres_config,
    get_whisper_config,
    get_embedding_config,
    validate_config,
    print_config_summary
)

# Get configurations
postgres_config = get_postgres_config()
whisper_config = get_whisper_config()
embedding_config = get_embedding_config()

# Validate and display
if validate_config():
    print_config_summary()
```

## 🔍 **Debugging and Troubleshooting**

### **Common Issues**
1. **Database Connection**: Check PostgreSQL is running
2. **Audio Recording**: Check microphone permissions
3. **Whisper Models**: Check internet connection for model download
4. **Embeddings**: Check if embeddings are being generated

### **Debug Commands**
```python
# Test database connection
python -c "from audio_storage_system import AudioStorageSystem; AudioStorageSystem(storage_backend='postgres')"

# Test audio recording
python test_audio.py

# Check embeddings
from embedding_queries import EmbeddingQueryEngine
stats = EmbeddingQueryEngine().get_embedding_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")
```

## 📊 **Data Flow Understanding**

### **Recording Flow**
1. **Audio Capture** → `sounddevice` records audio
2. **File Storage** → Audio saved to disk
3. **Transcription** → Whisper generates text
4. **Embedding** → Vector representation created
5. **Database** → All data stored in PostgreSQL

### **Search Flow**
1. **Query Input** → Text or audio query
2. **Embedding** → Query converted to vector
3. **Similarity** → Cosine similarity calculation
4. **Results** → Ranked similar content returned

### **Export Flow**
1. **Data Selection** → Choose data to export
2. **Format** → CSV, JSON, or numpy format
3. **File** → Save to disk with metadata

## 🎯 **Key Classes and Methods**

### **AudioStorageSystem**
```python
class AudioStorageSystem:
    def __init__(self, storage_backend="postgres", postgres_config=None)
    def record_and_store_audio(self, duration, whisper_model, generate_embeddings=True, metadata=None)
    def get_audio_info(self, audio_id)
    def search_similar_audio(self, query_text, limit=10, threshold=0.7)
    def list_all_audio(self, limit=None, offset=0)
```

### **EmbeddingQueryEngine**
```python
class EmbeddingQueryEngine:
    def __init__(self)
    def search_by_text_similarity(self, query_text, limit=10)
    def get_embeddings_by_metadata(self, metadata_key, metadata_value)
    def analyze_embedding(self, embedding_id)
    def export_embeddings_to_numpy(self, filename)
```

### **Configuration Management**
```python
# config.py provides centralized configuration
DEFAULT_POSTGRES_CONFIG = {...}
DEFAULT_WHISPER_MODEL = 'base'
DEFAULT_EMBEDDING_MODEL = 'text-embedding-ada-002'
```

## 🔧 **Extension Points**

### **Custom Embedding Models**
```python
class CustomAudioStorage(AudioStorageSystem):
    def generate_text_embedding(self, text):
        # Implement custom embedding logic
        return custom_embedding_function(text)
```

### **Custom Similarity Metrics**
```python
class CustomEmbeddingQueryEngine(EmbeddingQueryEngine):
    def calculate_cosine_similarity(self, embedding1, embedding2):
        # Implement custom similarity calculation
        return custom_similarity_function(embedding1, embedding2)
```

### **Custom Metadata Validation**
```python
def validate_metadata(metadata):
    # Implement custom validation logic
    required_keys = ['user', 'session']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required key: {key}")
    return True
```

## 📈 **Performance Considerations**

### **Database Optimization**
```sql
-- Create indexes for better performance
CREATE INDEX idx_audio_files_created_at ON audio_files(created_at);
CREATE INDEX idx_transcripts_confidence ON transcripts(confidence);
CREATE INDEX idx_metadata_key_value ON metadata(key, value);

-- Analyze tables
ANALYZE audio_files;
ANALYZE transcripts;
ANALYZE embeddings;
ANALYZE metadata;
```

### **Memory Management**
```python
# Monitor memory usage
import psutil
memory_percent = psutil.virtual_memory().percent

# Use generators for large datasets
def audio_file_generator():
    storage = AudioStorageSystem()
    for audio in storage.list_all_audio():
        yield audio
```

### **Batch Processing**
```python
# Process multiple files efficiently
def batch_process_audio_files(file_paths):
    storage = AudioStorageSystem()
    results = []
    
    for file_path in file_paths:
        try:
            audio_id = storage.store_audio_file(file_path)
            results.append({'file': file_path, 'id': audio_id, 'status': 'success'})
        except Exception as e:
            results.append({'file': file_path, 'error': str(e), 'status': 'failed'})
    
    return results
```

## 🎯 **Real-World Use Cases**

### **Meeting Transcription System**
```python
class MeetingTranscriptionSystem:
    def start_meeting_recording(self, meeting_id, participants):
        metadata = {
            'meeting_id': meeting_id,
            'participants': ','.join(participants),
            'session_type': 'meeting'
        }
        return self.storage.record_and_store_audio(
            duration=3600,  # 1 hour
            whisper_model="medium",
            generate_embeddings=True,
            metadata=metadata
        )
```

### **Voice Note Taking System**
```python
class VoiceNoteSystem:
    def create_voice_note(self, title, category, tags=None):
        metadata = {
            'title': title,
            'category': category,
            'tags': tags or '',
            'type': 'voice_note'
        }
        return self.storage.record_and_store_audio(
            duration=60,
            whisper_model="base",
            generate_embeddings=True,
            metadata=metadata
        )
```

## 🔮 **Future Enhancements**

### **Immediate Improvements**
1. **Real Embeddings**: Replace dummy embeddings with actual models
2. **Vector Database**: Add pgvector for native vector operations
3. **Web Interface**: Create browser-based recording interface
4. **API Endpoints**: Add RESTful API for remote access

### **Advanced Features**
1. **Multi-modal Embeddings**: Audio + text combined embeddings
2. **Real-time Processing**: Streaming audio transcription
3. **Cloud Storage**: S3/Google Cloud integration
4. **Advanced Search**: Semantic search with multiple modalities

## 📚 **Documentation References**

### **Essential Reading Order**
1. **README.md** - Project overview
2. **PROJECT_SETUP.md** - Complete setup guide
3. **DATABASE_SETUP.md** - Database configuration
4. **API_REFERENCE.md** - Full API documentation
5. **EXAMPLES.md** - Usage examples
6. **TROUBLESHOOTING.md** - Common issues

### **Quick Reference**
- **Setup**: `./setup.sh` → `createdb audio_storage_db` → `python audio_storage_system.py`
- **Recording**: `python interactive_recorder.py`
- **Search**: `python embedding_queries.py`
- **Demo**: `python postgres_demo.py`

## 🎉 **Success Metrics**

### **System Health Indicators**
- ✅ Database connection successful
- ✅ Audio recording working
- ✅ Whisper transcription accurate
- ✅ Embeddings generated
- ✅ Similarity search functional
- ✅ Data export working

### **Performance Benchmarks**
- **Recording**: < 5 seconds setup time
- **Transcription**: < 30 seconds for 1-minute audio
- **Search**: < 100ms for similarity queries
- **Export**: < 1 second for typical datasets

---

**This guide ensures LLM editors can quickly understand, set up, and extend the WhisperPOC system effectively. The project is designed to be self-documenting and LLM-friendly with comprehensive examples and clear architecture.** 