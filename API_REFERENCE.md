# WhisperPOC API Reference

## 📚 **Overview**

This document provides a comprehensive API reference for the WhisperPOC audio storage system. It covers all classes, methods, parameters, and usage examples.

## 🏗️ **Core Classes**

### **AudioStorageSystem**

The main class for audio recording, storage, and management.

#### **Constructor**
```python
AudioStorageSystem(
    storage_backend="postgres",
    postgres_config=None,
    audio_storage_dir="audio_storage"
)
```

**Parameters:**
- `storage_backend` (str): Storage backend type ("postgres", "sqlite", "json")
- `postgres_config` (dict): PostgreSQL configuration dictionary
- `audio_storage_dir` (str): Directory for storing audio files

**Example:**
```python
from audio_storage_system import AudioStorageSystem

# Initialize with PostgreSQL
storage = AudioStorageSystem(
    storage_backend="postgres",
    postgres_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'audio_storage_db',
        'user': 'postgres',
        'password': ''
    }
)
```

#### **Core Methods**

##### **record_and_store_audio()**
Records audio and stores it in the database with transcription and embeddings.

```python
audio_id = storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True,
    metadata=None,
    sample_rate=16000,
    channels=1
)
```

**Parameters:**
- `duration` (int): Recording duration in seconds
- `whisper_model` (str): Whisper model to use ("tiny", "base", "small", "medium", "large")
- `generate_embeddings` (bool): Whether to generate embeddings
- `metadata` (dict): Optional metadata dictionary
- `sample_rate` (int): Audio sample rate
- `channels` (int): Number of audio channels

**Returns:** `int` - Audio file ID

##### **store_audio_file()**
Stores an existing audio file in the database.

```python
audio_id = storage.store_audio_file(
    file_path,
    generate_transcript=True,
    generate_embeddings=True,
    metadata=None
)
```

**Parameters:**
- `file_path` (str): Path to audio file
- `generate_transcript` (bool): Whether to generate transcript
- `generate_embeddings` (bool): Whether to generate embeddings
- `metadata` (dict): Optional metadata

**Returns:** `int` - Audio file ID

##### **get_audio_info()**
Retrieves complete information about an audio file.

```python
audio_info = storage.get_audio_info(audio_id)
```

**Parameters:**
- `audio_id` (int): Audio file ID

**Returns:** `dict` - Complete audio information including file, transcript, embeddings, and metadata

##### **search_similar_audio()**
Searches for similar audio based on transcript content.

```python
results = storage.search_similar_audio(
    query_text,
    limit=10,
    threshold=0.7
)
```

**Parameters:**
- `query_text` (str): Search query text
- `limit` (int): Maximum number of results
- `threshold` (float): Similarity threshold

**Returns:** `list` - List of similar audio files

##### **list_all_audio()**
Lists all audio files in the database.

```python
audio_files = storage.list_all_audio(
    limit=None,
    offset=0,
    order_by="created_at",
    order_direction="DESC"
)
```

**Parameters:**
- `limit` (int): Maximum number of results
- `offset` (int): Number of results to skip
- `order_by` (str): Field to order by
- `order_direction` (str): Order direction ("ASC" or "DESC")

**Returns:** `list` - List of audio file information

#### **Database Management Methods**

##### **get_audio_count()**
```python
count = storage.get_audio_count()
```

**Returns:** `int` - Total number of audio files

##### **get_transcript_count()**
```python
count = storage.get_transcript_count()
```

**Returns:** `int` - Total number of transcripts

##### **get_embedding_count()**
```python
count = storage.get_embedding_count()
```

**Returns:** `int` - Total number of embeddings

##### **cleanup_old_files()**
Removes old audio files and database records.

```python
removed_count = storage.cleanup_old_files(
    days_old=30,
    keep_embeddings=True
)
```

**Parameters:**
- `days_old` (int): Age threshold in days
- `keep_embeddings` (bool): Whether to keep embeddings

**Returns:** `int` - Number of files removed

---

### **EmbeddingQueryEngine**

Advanced query engine for vector embeddings and similarity search.

#### **Constructor**
```python
query_engine = EmbeddingQueryEngine()
```

#### **Core Methods**

##### **get_all_embeddings()**
Retrieves all embeddings of a specific type.

```python
embeddings = query_engine.get_all_embeddings(embedding_type="transcript")
```

**Parameters:**
- `embedding_type` (str): Type of embeddings to retrieve

**Returns:** `list` - List of embedding data

##### **analyze_embedding()**
Analyzes a specific embedding and provides statistical information.

```python
analysis = query_engine.analyze_embedding(embedding_id)
```

**Parameters:**
- `embedding_id` (int): Embedding ID

**Returns:** `tuple` - (embedding_data, embedding_array)

##### **find_similar_embeddings()**
Finds similar embeddings using cosine similarity.

```python
similar_embeddings = query_engine.find_similar_embeddings(
    query_embedding,
    limit=10,
    embedding_type="transcript"
)
```

**Parameters:**
- `query_embedding` (numpy.ndarray): Query embedding vector
- `limit` (int): Maximum number of results
- `embedding_type` (str): Type of embeddings to search

**Returns:** `list` - List of similar embeddings

##### **search_by_text_similarity()**
Searches for similar content using text queries.

```python
results = query_engine.search_by_text_similarity(
    query_text,
    limit=10
)
```

**Parameters:**
- `query_text` (str): Text query
- `limit` (int): Maximum number of results

**Returns:** `list` - List of similar audio files

##### **get_embeddings_by_metadata()**
Filters embeddings based on metadata.

```python
embeddings = query_engine.get_embeddings_by_metadata(
    metadata_key="user",
    metadata_value="john"
)
```

**Parameters:**
- `metadata_key` (str): Metadata key to filter by
- `metadata_value` (str): Metadata value to match

**Returns:** `list` - List of filtered embeddings

##### **get_embedding_statistics()**
Gets comprehensive statistics about embeddings.

```python
stats = query_engine.get_embedding_statistics()
```

**Returns:** `dict` - Embedding statistics

##### **export_embeddings_to_numpy()**
Exports embeddings to numpy format.

```python
export_file = query_engine.export_embeddings_to_numpy("embeddings.npz")
```

**Parameters:**
- `filename` (str): Output filename

**Returns:** `str` - Path to exported file

##### **load_embeddings_from_numpy()**
Loads embeddings from numpy format.

```python
embeddings, metadata = query_engine.load_embeddings_from_numpy("embeddings.npz")
```

**Parameters:**
- `filename` (str): Input filename

**Returns:** `tuple` - (embeddings, metadata)

#### **Utility Methods**

##### **calculate_cosine_similarity()**
```python
similarity = query_engine.calculate_cosine_similarity(embedding1, embedding2)
```

**Parameters:**
- `embedding1` (numpy.ndarray): First embedding vector
- `embedding2` (numpy.ndarray): Second embedding vector

**Returns:** `float` - Cosine similarity score

##### **get_embedding_array()**
```python
embedding_array = query_engine.get_embedding_array(embedding_id)
```

**Parameters:**
- `embedding_id` (int): Embedding ID

**Returns:** `numpy.ndarray` - Embedding vector

---

### **InteractiveRecorder**

Interactive command-line interface for audio recording.

#### **Constructor**
```python
recorder = InteractiveRecorder()
```

#### **Core Methods**

##### **start_interactive_session()**
Starts an interactive recording session.

```python
recorder.start_interactive_session()
```

##### **record_audio()**
Records audio with specified parameters.

```python
audio_id = recorder.record_audio(
    duration=10,
    whisper_model="base",
    metadata=None
)
```

**Parameters:**
- `duration` (int): Recording duration
- `whisper_model` (str): Whisper model
- `metadata` (dict): Optional metadata

**Returns:** `int` - Audio file ID

---

## 🎤 **Audio Processing Classes**

### **WhisperLiveTranscriber**

Real-time audio transcription with Voice Activity Detection.

#### **Constructor**
```python
transcriber = WhisperLiveTranscriber(
    model_name="base",
    sample_rate=16000,
    chunk_duration_ms=30
)
```

**Parameters:**
- `model_name` (str): Whisper model name
- `sample_rate` (int): Audio sample rate
- `chunk_duration_ms` (int): Chunk duration in milliseconds

#### **Core Methods**

##### **start()**
Starts real-time transcription.

```python
transcriber.start()
```

##### **stop()**
Stops real-time transcription.

```python
transcriber.stop()
```

##### **is_speech()**
Detects if audio contains speech.

```python
speech_detected = transcriber.is_speech(audio_chunk)
```

**Parameters:**
- `audio_chunk` (bytes): Audio chunk data

**Returns:** `bool` - True if speech detected

---

### **SimpleWhisperTranscriber**

Simple fixed-duration audio transcription.

#### **Constructor**
```python
transcriber = SimpleWhisperTranscriber(
    model_name="base",
    sample_rate=16000
)
```

#### **Core Methods**

##### **start_continuous()**
Starts continuous recording and transcription.

```python
transcriber.start_continuous(record_duration=5)
```

**Parameters:**
- `record_duration` (int): Recording duration in seconds

##### **record_audio()**
Records audio for specified duration.

```python
audio_data = transcriber.record_audio(duration=10)
```

**Parameters:**
- `duration` (int): Recording duration

**Returns:** `numpy.ndarray` - Audio data

---

## 🤖 **Voice-to-LLM Classes**

### **VoiceToLLM**

Cloud-based voice-to-LLM integration.

#### **Constructor**
```python
voice_llm = VoiceToLLM(
    openai_api_key="your_key",
    anthropic_api_key="your_key",
    google_api_key="your_key"
)
```

#### **Core Methods**

##### **start_interactive_session()**
Starts interactive voice-to-LLM session.

```python
voice_llm.start_interactive_session()
```

##### **call_openai_gpt4o()**
Calls OpenAI GPT-4o with audio input.

```python
response = voice_llm.call_openai_gpt4o(audio_base64, prompt="")
```

**Parameters:**
- `audio_base64` (str): Base64-encoded audio
- `prompt` (str): Optional text prompt

**Returns:** `str` - LLM response

##### **call_anthropic_claude()**
Calls Anthropic Claude with audio input.

```python
response = voice_llm.call_anthropic_claude(audio_base64, prompt="")
```

##### **call_google_gemini()**
Calls Google Gemini with audio input.

```python
response = voice_llm.call_google_gemini(audio_base64, prompt="")
```

---

### **LocalVoiceToLLM**

Local voice-to-LLM integration using Whisper + local LLMs.

#### **Constructor**
```python
local_voice_llm = LocalVoiceToLLM(
    ollama_base_url="http://localhost:11434",
    localai_base_url="http://localhost:8080"
)
```

#### **Core Methods**

##### **call_ollama_llm()**
Calls local Ollama LLM.

```python
response = local_voice_llm.call_ollama_llm(
    transcript,
    model="llama2",
    system_prompt=""
)
```

##### **call_localai_llm()**
Calls local LocalAI server.

```python
response = local_voice_llm.call_localai_llm(
    transcript,
    model="gpt-3.5-turbo",
    system_prompt=""
)
```

---

## ⚙️ **Configuration**

### **config.py**

Central configuration management.

#### **Configuration Functions**

##### **get_postgres_config()**
```python
config = get_postgres_config()
```

**Returns:** `dict` - PostgreSQL configuration

##### **get_whisper_config()**
```python
config = get_whisper_config()
```

**Returns:** `dict` - Whisper configuration

##### **get_embedding_config()**
```python
config = get_embedding_config()
```

**Returns:** `dict` - Embedding configuration

##### **get_audio_config()**
```python
config = get_audio_config()
```

**Returns:** `dict` - Audio configuration

##### **validate_config()**
```python
is_valid = validate_config()
```

**Returns:** `bool` - Configuration validity

##### **print_config_summary()**
```python
print_config_summary()
```

Prints current configuration summary.

---

## 📊 **Data Structures**

### **Audio File Information**
```python
{
    'id': 1,
    'file_hash': 'abc123...',
    'file_path': '/path/to/audio.wav',
    'file_size': 1024000,
    'duration': 10.5,
    'sample_rate': 16000,
    'channels': 1,
    'created_at': '2024-01-01 12:00:00',
    'updated_at': '2024-01-01 12:00:00'
}
```

### **Transcript Information**
```python
{
    'id': 1,
    'audio_file_id': 1,
    'transcription': 'Hello, this is a test recording.',
    'confidence': 0.95,
    'language': 'en',
    'created_at': '2024-01-01 12:00:00'
}
```

### **Embedding Information**
```python
{
    'id': 1,
    'audio_file_id': 1,
    'transcript_id': 1,
    'embedding_type': 'transcript',
    'embedding_data': b'...',  # Pickled numpy array
    'embedding_dimension': 1536,
    'created_at': '2024-01-01 12:00:00'
}
```

### **Metadata Information**
```python
{
    'id': 1,
    'audio_file_id': 1,
    'key': 'user',
    'value': 'john',
    'created_at': '2024-01-01 12:00:00'
}
```

---

## 🔧 **Error Handling**

### **Common Exceptions**

#### **AudioStorageError**
Raised for audio storage-related errors.

```python
try:
    storage.record_and_store_audio(duration=10)
except AudioStorageError as e:
    print(f"Storage error: {e}")
```

#### **DatabaseError**
Raised for database-related errors.

```python
try:
    storage.get_audio_info(1)
except DatabaseError as e:
    print(f"Database error: {e}")
```

#### **AudioProcessingError**
Raised for audio processing errors.

```python
try:
    transcriber.start()
except AudioProcessingError as e:
    print(f"Audio processing error: {e}")
```

---

## 📝 **Usage Examples**

### **Basic Recording and Storage**
```python
from audio_storage_system import AudioStorageSystem

# Initialize storage
storage = AudioStorageSystem(storage_backend="postgres")

# Record and store audio
audio_id = storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True,
    metadata={"user": "john", "session": "meeting"}
)

# Get audio information
audio_info = storage.get_audio_info(audio_id)
print(f"Transcript: {audio_info['transcript']['transcription']}")
```

### **Similarity Search**
```python
from embedding_queries import EmbeddingQueryEngine

# Initialize query engine
query_engine = EmbeddingQueryEngine()

# Search for similar content
results = query_engine.search_by_text_similarity(
    query_text="meeting discussion",
    limit=5
)

for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Transcript: {result['transcription']}")
```

### **Batch Processing**
```python
from audio_storage_system import AudioStorageSystem

storage = AudioStorageSystem(storage_backend="postgres")

# Record multiple files
for i in range(5):
    audio_id = storage.record_and_store_audio(
        duration=5,
        whisper_model="tiny",
        metadata={"batch": "test", "number": i}
    )
    print(f"Recorded audio {i+1}: {audio_id}")

# List all recordings
audio_files = storage.list_all_audio(limit=10)
for audio in audio_files:
    print(f"File: {audio['file_path']}")
```

### **Export and Analysis**
```python
from embedding_queries import EmbeddingQueryEngine

query_engine = EmbeddingQueryEngine()

# Export embeddings
export_file = query_engine.export_embeddings_to_numpy("embeddings.npz")

# Get statistics
stats = query_engine.get_embedding_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")

# Analyze specific embedding
analysis = query_engine.analyze_embedding(1)
if analysis:
    embedding_array = analysis[1]
    print(f"Embedding shape: {embedding_array.shape}")
```

---

## 🔮 **Advanced Features**

### **Custom Embedding Generation**
```python
def custom_embedding_function(text):
    # Implement your own embedding generation
    return numpy.random.rand(1536)  # Dummy embedding

# Override in AudioStorageSystem
storage.generate_text_embedding = custom_embedding_function
```

### **Custom Metadata Validation**
```python
def validate_metadata(metadata):
    required_keys = ['user', 'session']
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: {key}")
    return True

# Use in recording
storage.record_and_store_audio(
    duration=10,
    metadata={"user": "john", "session": "meeting"},
    metadata_validator=validate_metadata
)
```

### **Custom Similarity Metrics**
```python
def custom_similarity(embedding1, embedding2):
    # Implement custom similarity metric
    return numpy.dot(embedding1, embedding2)

# Override in EmbeddingQueryEngine
query_engine.calculate_cosine_similarity = custom_similarity
```

---

This API reference provides comprehensive documentation for all WhisperPOC components. For additional examples and tutorials, see the `examples/` directory and other documentation files. 