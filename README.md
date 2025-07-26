# WhisperPOC - Audio Storage System with PostgreSQL

A comprehensive audio recording, transcription, and storage system using OpenAI Whisper and PostgreSQL with vector embeddings for similarity search.

## 🎯 **Features**

- **🎤 Real-time Audio Recording** - Record audio with custom duration and metadata
- **🎯 Whisper Transcription** - Automatic speech-to-text using OpenAI Whisper
- **🗄️ PostgreSQL Storage** - Enterprise-grade database with full ACID compliance
- **🧠 Vector Embeddings** - Store and query embeddings for similarity search
- **🔍 Advanced Search** - Semantic search using cosine similarity
- **📊 Metadata Management** - Flexible tagging and organization system
- **📤 Export Capabilities** - Export data to CSV and numpy formats
- **🔄 Interactive Recording** - User-friendly recording interface

## 🏗️ **Architecture**

```
WhisperPOC/
├── audio_storage_system.py      # Main storage system
├── interactive_recorder.py      # Interactive recording interface
├── embedding_queries.py         # Vector embedding query engine
├── postgres_demo.py            # PostgreSQL demonstration
├── requirements.txt            # Python dependencies
├── setup.sh                   # Automated setup script
├── README.md                  # This file
├── AUDIO_STORAGE_GUIDE.md     # Comprehensive storage guide
├── EMBEDDING_QUERY_GUIDE.md   # Vector embedding guide
├── POSTGRES_SUMMARY.md        # PostgreSQL implementation summary
└── LOCAL_VOICE_TO_LLM_GUIDE.md # Local voice-to-LLM guide
```

## 🚀 **Quick Start**

### **1. Prerequisites**
- Python 3.8+
- PostgreSQL 12+
- macOS/Linux/Windows

### **2. Installation**
```bash
# Clone the repository
git clone https://github.com/winkidzz/whisperpoc.git
cd whisperpoc

# Run automated setup
chmod +x setup.sh
./setup.sh

# Or manual setup
pip install -r requirements.txt
```

### **3. Database Setup**
```bash
# Create PostgreSQL database
createdb audio_storage_db

# The system will automatically create tables on first run
```

### **4. Start Recording**
```bash
# Interactive recording session
python interactive_recorder.py

# Or basic recording
python audio_storage_system.py
```

## 📖 **Usage Examples**

### **Interactive Recording**
```python
from interactive_recorder import interactive_recorder

# Start interactive recording session
interactive_recorder()
```

### **Programmatic Recording**
```python
from audio_storage_system import AudioStorageSystem

# Initialize PostgreSQL storage
storage = AudioStorageSystem(storage_backend="postgres")

# Record and store audio
audio_id = storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True,
    metadata={"user": "john", "session": "meeting_1"}
)
```

### **Vector Embedding Queries**
```python
from embedding_queries import EmbeddingQueryEngine

# Initialize query engine
query_engine = EmbeddingQueryEngine()

# Find similar audio
similar_embeddings = query_engine.search_by_text_similarity(
    query_text="meeting discussion",
    limit=5
)

# Export embeddings
export_file = query_engine.export_embeddings_to_numpy("embeddings.npz")
```

## 🗄️ **Database Schema**

### **Tables**
- **`audio_files`** - Audio file metadata and storage information
- **`transcripts`** - Whisper transcriptions with confidence scores
- **`embeddings`** - Vector embeddings for similarity search
- **`metadata`** - Flexible key-value storage for additional information

### **Sample Queries**
```sql
-- View all audio files with transcripts
SELECT af.file_path, t.transcription, af.duration
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
ORDER BY af.created_at DESC;

-- Find embeddings by user
SELECT e.id, t.transcription, m.value as user
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
JOIN metadata m ON e.audio_file_id = m.audio_file_id
WHERE m.key = 'user' AND m.value = 'john';
```

## 🔧 **Configuration**

### **PostgreSQL Configuration**
```python
postgres_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'audio_storage_db',
    'user': 'your_username',
    'password': 'your_password'
}

storage = AudioStorageSystem(
    storage_backend="postgres",
    postgres_config=postgres_config
)
```

### **Whisper Models**
- **tiny** - Fastest, least accurate
- **base** - Good balance (default)
- **small** - Better accuracy
- **medium** - High accuracy
- **large** - Best accuracy, slowest

## 📊 **Performance**

### **Current Database Status**
- **Audio Files:** 11 recordings
- **Total Size:** 3.54 MB
- **Average Duration:** 5.3 seconds
- **Embeddings:** 3 transcript embeddings (1536 dimensions each)
- **Storage:** 0.04 MB for embeddings

### **Query Performance**
- **Similarity Search:** < 100ms for small datasets
- **Metadata Filtering:** < 50ms with proper indexing
- **Export Operations:** < 1s for typical datasets

## 🔍 **Search Capabilities**

### **1. Similarity Search**
```python
# Find similar audio based on transcript content
results = storage.search_similar_audio("meeting discussion", limit=10)
```

### **2. Metadata Filtering**
```python
# Filter by user, session, priority, etc.
user_audio = query_engine.get_embeddings_by_metadata("user", "john")
```

### **3. Time-based Queries**
```sql
-- Get recent recordings
SELECT * FROM audio_files 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;
```

## 📤 **Export Features**

### **CSV Export**
```python
# Export all audio data to CSV
storage.cursor.execute("""
    SELECT af.*, t.transcription, t.confidence
    FROM audio_files af
    LEFT JOIN transcripts t ON af.id = t.audio_file_id
    ORDER BY af.created_at DESC
""")
```

### **Numpy Export**
```python
# Export embeddings to numpy format
export_file = query_engine.export_embeddings_to_numpy("embeddings.npz")
embeddings, metadata = query_engine.load_embeddings_from_numpy("embeddings.npz")
```

## 🛠️ **Development**

### **Running Tests**
```bash
# Test audio recording
python test_audio.py

# Test PostgreSQL connection
python postgres_demo.py

# Test embedding queries
python embedding_queries.py
```

### **Adding New Features**
1. **New Storage Backend** - Extend `AudioStorageSystem` class
2. **New Embedding Models** - Implement in `generate_text_embedding()`
3. **New Search Methods** - Add to `EmbeddingQueryEngine` class

## 📚 **Documentation**

- **[AUDIO_STORAGE_GUIDE.md](AUDIO_STORAGE_GUIDE.md)** - Comprehensive storage guide
- **[EMBEDDING_QUERY_GUIDE.md](EMBEDDING_QUERY_GUIDE.md)** - Vector embedding operations
- **[POSTGRES_SUMMARY.md](POSTGRES_SUMMARY.md)** - PostgreSQL implementation details
- **[LOCAL_VOICE_TO_LLM_GUIDE.md](LOCAL_VOICE_TO_LLM_GUIDE.md)** - Local voice-to-LLM integration

## 🔮 **Future Enhancements**

- **Real-time Processing** - Streaming audio transcription
- **Advanced Embeddings** - Multi-modal embeddings (audio + text)
- **Vector Database Integration** - ChromaDB, Pinecone, Weaviate
- **Cloud Storage** - S3, Google Cloud Storage integration
- **API Endpoints** - RESTful API for remote access
- **Web Interface** - Browser-based recording and search

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **OpenAI Whisper** - Speech recognition model
- **PostgreSQL** - Database system
- **NumPy** - Numerical computing
- **SoundDevice** - Audio recording

---

**Built with ❤️ for audio processing and AI applications** 