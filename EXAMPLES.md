# WhisperPOC Examples Guide

## 🎯 **Overview**

This guide provides comprehensive examples of how to use the WhisperPOC audio storage system for various scenarios and use cases.

## 🚀 **Quick Start Examples**

### **1. Basic Audio Recording**
```python
from audio_storage_system import AudioStorageSystem

# Initialize storage system
storage = AudioStorageSystem(storage_backend="postgres")

# Record 10 seconds of audio
audio_id = storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True
)

print(f"Recorded audio with ID: {audio_id}")
```

### **2. Interactive Recording Session**
```python
from interactive_recorder import interactive_recorder

# Start interactive recording
interactive_recorder()
```

### **3. Voice-to-LLM Conversation**
```python
from voice_to_llm import VoiceToLLM

# Initialize voice-to-LLM system
voice_llm = VoiceToLLM(
    openai_api_key="your_openai_key"
)

# Start interactive session
voice_llm.start_interactive_session()
```

## 📊 **Data Management Examples**

### **4. Batch Recording with Metadata**
```python
from audio_storage_system import AudioStorageSystem
import time

storage = AudioStorageSystem(storage_backend="postgres")

# Record multiple audio files with different metadata
recording_scenarios = [
    {
        'duration': 5,
        'metadata': {'user': 'john', 'session': 'quick_notes', 'priority': 'low'}
    },
    {
        'duration': 30,
        'metadata': {'user': 'john', 'session': 'meeting', 'priority': 'high'}
    },
    {
        'duration': 60,
        'metadata': {'user': 'john', 'session': 'interview', 'priority': 'high'}
    }
]

for i, scenario in enumerate(recording_scenarios, 1):
    print(f"Recording scenario {i}...")
    
    audio_id = storage.record_and_store_audio(
        duration=scenario['duration'],
        whisper_model="base",
        generate_embeddings=True,
        metadata=scenario['metadata']
    )
    
    print(f"Recorded audio {i}: {audio_id}")
    time.sleep(2)  # Brief pause between recordings
```

### **5. Import Existing Audio Files**
```python
from audio_storage_system import AudioStorageSystem
import os

storage = AudioStorageSystem(storage_backend="postgres")

# Import all WAV files from a directory
audio_directory = "existing_audio_files"
for filename in os.listdir(audio_directory):
    if filename.endswith('.wav'):
        file_path = os.path.join(audio_directory, filename)
        
        audio_id = storage.store_audio_file(
            file_path=file_path,
            generate_transcript=True,
            generate_embeddings=True,
            metadata={
                'source': 'imported',
                'original_filename': filename,
                'user': 'john'
            }
        )
        
        print(f"Imported {filename}: {audio_id}")
```

### **6. Search and Filter Audio**
```python
from audio_storage_system import AudioStorageSystem

storage = AudioStorageSystem(storage_backend="postgres")

# List all audio files
all_audio = storage.list_all_audio(limit=50)

# Search for similar content
similar_results = storage.search_similar_audio(
    query_text="meeting discussion",
    limit=10,
    threshold=0.7
)

print(f"Found {len(similar_results)} similar recordings")

# Filter by metadata
from embedding_queries import EmbeddingQueryEngine
query_engine = EmbeddingQueryEngine()

user_recordings = query_engine.get_embeddings_by_metadata(
    metadata_key="user",
    metadata_value="john"
)

print(f"Found {len(user_recordings)} recordings by user 'john'")
```

## 🔍 **Analysis and Query Examples**

### **7. Embedding Analysis**
```python
from embedding_queries import EmbeddingQueryEngine
import numpy as np

query_engine = EmbeddingQueryEngine()

# Get embedding statistics
stats = query_engine.get_embedding_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Average dimension: {stats['avg_dimension']:.1f}")

# Analyze specific embeddings
all_embeddings = query_engine.get_all_embeddings('transcript')

for i, embedding_data in enumerate(all_embeddings[:5], 1):
    print(f"\nEmbedding {i}:")
    analysis = query_engine.analyze_embedding(embedding_data['id'])
    if analysis:
        embedding_array = analysis[1]
        print(f"  Shape: {embedding_array.shape}")
        print(f"  Mean: {np.mean(embedding_array):.4f}")
        print(f"  Std: {np.std(embedding_array):.4f}")
        print(f"  Transcript: {embedding_data['transcription'][:50]}...")
```

### **8. Similarity Clustering**
```python
from embedding_queries import EmbeddingQueryEngine

query_engine = EmbeddingQueryEngine()

# Get all embeddings
all_embeddings = query_engine.get_all_embeddings('transcript')

# Calculate similarity matrix
similarities = []
for i, emb1 in enumerate(all_embeddings):
    for j, emb2 in enumerate(all_embeddings[i+1:], i+1):
        similarity = query_engine.calculate_cosine_similarity(
            query_engine.get_embedding_array(emb1['id']),
            query_engine.get_embedding_array(emb2['id'])
        )
        similarities.append({
            'emb1': emb1['transcription'][:30],
            'emb2': emb2['transcription'][:30],
            'similarity': similarity
        })

# Sort by similarity
similarities.sort(key=lambda x: x['similarity'], reverse=True)

print("Top similar pairs:")
for i, pair in enumerate(similarities[:5], 1):
    print(f"{i}. Similarity: {pair['similarity']:.3f}")
    print(f"   '{pair['emb1']}'")
    print(f"   '{pair['emb2']}'")
    print()
```

### **9. Advanced Database Queries**
```python
from audio_storage_system import AudioStorageSystem

storage = AudioStorageSystem(storage_backend="postgres")

# Get high-confidence transcriptions
storage.cursor.execute("""
    SELECT af.file_path, t.transcription, t.confidence
    FROM audio_files af
    JOIN transcripts t ON af.id = t.audio_file_id
    WHERE t.confidence > 0.9
    ORDER BY t.confidence DESC
    LIMIT 10
""")

high_confidence = storage.cursor.fetchall()
print(f"Found {len(high_confidence)} high-confidence transcriptions")

# Get recent recordings by user
storage.cursor.execute("""
    SELECT af.file_path, t.transcription, m.value as user, af.created_at
    FROM audio_files af
    JOIN transcripts t ON af.id = t.audio_file_id
    JOIN metadata m ON af.id = m.audio_file_id
    WHERE m.key = 'user' AND m.value = 'john'
    ORDER BY af.created_at DESC
    LIMIT 10
""")

user_recordings = storage.cursor.fetchall()
print(f"Found {len(user_recordings)} recent recordings by user")
```

## 🤖 **Voice-to-LLM Examples**

### **10. Cloud Voice-to-LLM**
```python
from voice_to_llm import VoiceToLLM

# Initialize with API keys
voice_llm = VoiceToLLM(
    openai_api_key="your_openai_key",
    anthropic_api_key="your_anthropic_key",
    google_api_key="your_google_key"
)

# Single voice query
def ask_question(question):
    # Record audio
    audio_base64 = voice_llm.record_audio(duration=10)
    
    # Get response from different providers
    openai_response = voice_llm.call_openai_gpt4o(audio_base64, question)
    claude_response = voice_llm.call_anthropic_claude(audio_base64, question)
    gemini_response = voice_llm.call_google_gemini(audio_base64, question)
    
    return {
        'openai': openai_response,
        'claude': claude_response,
        'gemini': gemini_response
    }

# Use the function
responses = ask_question("What's the weather like today?")
for provider, response in responses.items():
    print(f"{provider.upper()}: {response}")
```

### **11. Local Voice-to-LLM**
```python
from local_voice_to_llm import LocalVoiceToLLM

# Initialize local system
local_voice_llm = LocalVoiceToLLM()

# Check available providers
ollama_status = local_voice_llm.check_llm_provider('ollama')
localai_status = local_voice_llm.check_llm_provider('localai')

print(f"Ollama available: {ollama_status}")
print(f"LocalAI available: {localai_status}")

# Use available provider
if ollama_status:
    transcript = local_voice_llm.transcribe_audio(duration=10)
    response = local_voice_llm.call_ollama_llm(
        transcript,
        model="llama2",
        system_prompt="You are a helpful assistant."
    )
    print(f"Response: {response}")
```

## 📈 **Performance and Monitoring Examples**

### **12. Performance Monitoring**
```python
import time
import psutil
from audio_storage_system import AudioStorageSystem

def monitor_performance():
    storage = AudioStorageSystem(storage_backend="postgres")
    
    # Monitor system resources
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    print(f"System - CPU: {cpu_percent}%, Memory: {memory_percent}%")
    
    # Monitor database performance
    start_time = time.time()
    audio_count = storage.get_audio_count()
    query_time = time.time() - start_time
    
    print(f"Database - Audio files: {audio_count}, Query time: {query_time:.3f}s")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'audio_count': audio_count,
        'query_time': query_time
    }

# Monitor over time
for i in range(10):
    metrics = monitor_performance()
    time.sleep(5)
```

### **13. Batch Processing with Progress**
```python
from audio_storage_system import AudioStorageSystem
from tqdm import tqdm
import time

def batch_process_audio_files(file_paths):
    storage = AudioStorageSystem(storage_backend="postgres")
    results = []
    
    for file_path in tqdm(file_paths, desc="Processing audio files"):
        try:
            start_time = time.time()
            
            audio_id = storage.store_audio_file(
                file_path=file_path,
                generate_transcript=True,
                generate_embeddings=True
            )
            
            processing_time = time.time() - start_time
            results.append({
                'file_path': file_path,
                'audio_id': audio_id,
                'processing_time': processing_time,
                'status': 'success'
            })
            
        except Exception as e:
            results.append({
                'file_path': file_path,
                'error': str(e),
                'status': 'failed'
            })
    
    return results

# Use the function
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = batch_process_audio_files(audio_files)

# Summary
successful = sum(1 for r in results if r['status'] == 'success')
failed = sum(1 for r in results if r['status'] == 'failed')
print(f"Processed {successful} files successfully, {failed} failed")
```

## 📤 **Export and Integration Examples**

### **14. Export Data for Analysis**
```python
from embedding_queries import EmbeddingQueryEngine
import pandas as pd
from datetime import datetime

def export_data_for_analysis():
    query_engine = EmbeddingQueryEngine()
    
    # Export embeddings
    export_file = query_engine.export_embeddings_to_numpy("embeddings_export.npz")
    print(f"Embeddings exported to: {export_file}")
    
    # Export to CSV
    from audio_storage_system import AudioStorageSystem
    storage = AudioStorageSystem(storage_backend="postgres")
    
    storage.cursor.execute("""
        SELECT 
            af.id,
            af.file_path,
            af.duration,
            af.file_size,
            t.transcription,
            t.confidence,
            t.language,
            af.created_at
        FROM audio_files af
        LEFT JOIN transcripts t ON af.id = t.audio_file_id
        ORDER BY af.created_at DESC
    """)
    
    data = storage.cursor.fetchall()
    df = pd.DataFrame(data, columns=[
        'id', 'file_path', 'duration', 'file_size',
        'transcription', 'confidence', 'language', 'created_at'
    ])
    
    csv_filename = f"audio_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data exported to: {csv_filename}")
    
    return export_file, csv_filename

export_file, csv_file = export_data_for_analysis()
```

### **15. Integration with External Systems**
```python
import requests
from audio_storage_system import AudioStorageSystem

def integrate_with_external_api():
    storage = AudioStorageSystem(storage_backend="postgres")
    
    # Get recent recordings
    recent_audio = storage.list_all_audio(limit=5)
    
    for audio in recent_audio:
        audio_info = storage.get_audio_info(audio['id'])
        
        # Send to external API
        payload = {
            'audio_id': audio['id'],
            'transcription': audio_info['transcript']['transcription'],
            'confidence': audio_info['transcript']['confidence'],
            'metadata': audio_info['metadata'],
            'timestamp': audio_info['audio_file']['created_at']
        }
        
        try:
            response = requests.post(
                'https://api.example.com/audio-analysis',
                json=payload,
                headers={'Authorization': 'Bearer your_api_key'}
            )
            
            if response.status_code == 200:
                print(f"Successfully sent audio {audio['id']} to external API")
            else:
                print(f"Failed to send audio {audio['id']}: {response.status_code}")
                
        except Exception as e:
            print(f"Error sending audio {audio['id']}: {e}")

integrate_with_external_api()
```

## 🔧 **Customization Examples**

### **16. Custom Embedding Generation**
```python
from audio_storage_system import AudioStorageSystem
import numpy as np
from sentence_transformers import SentenceTransformer

class CustomAudioStorage(AudioStorageSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load custom embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_text_embedding(self, text):
        """Generate embeddings using SentenceTransformers."""
        if not text or text.strip() == "":
            return np.random.rand(384)  # Fallback for empty text
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.random.rand(384)  # Fallback

# Use custom storage system
custom_storage = CustomAudioStorage(storage_backend="postgres")

audio_id = custom_storage.record_and_store_audio(
    duration=10,
    whisper_model="base",
    generate_embeddings=True,
    metadata={'user': 'john', 'custom_embedding': 'sentence_transformers'}
)
```

### **17. Custom Metadata Validation**
```python
from audio_storage_system import AudioStorageSystem

def validate_metadata(metadata):
    """Custom metadata validation function."""
    required_keys = ['user', 'session']
    optional_keys = ['priority', 'tags', 'notes']
    
    # Check required keys
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata key: {key}")
    
    # Validate priority values
    if 'priority' in metadata:
        valid_priorities = ['low', 'normal', 'high']
        if metadata['priority'] not in valid_priorities:
            raise ValueError(f"Invalid priority: {metadata['priority']}")
    
    # Validate user format
    if 'user' in metadata:
        if not metadata['user'].isalnum():
            raise ValueError("User must be alphanumeric")
    
    return True

# Use custom validation
storage = AudioStorageSystem(storage_backend="postgres")

try:
    audio_id = storage.record_and_store_audio(
        duration=10,
        metadata={
            'user': 'john123',
            'session': 'meeting',
            'priority': 'high',
            'tags': 'important,urgent'
        }
    )
    print(f"Recording successful: {audio_id}")
except ValueError as e:
    print(f"Validation error: {e}")
```

### **18. Custom Similarity Metrics**
```python
from embedding_queries import EmbeddingQueryEngine
import numpy as np
from scipy.spatial.distance import euclidean, manhattan

class CustomEmbeddingQueryEngine(EmbeddingQueryEngine):
    def calculate_cosine_similarity(self, embedding1, embedding2):
        """Override with custom similarity calculation."""
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Apply custom weighting
        weighted_sim = cosine_sim * 0.7 + (1 - euclidean(embedding1, embedding2) / 100) * 0.3
        
        return max(0.0, min(1.0, weighted_sim))  # Clamp to [0, 1]

# Use custom query engine
custom_query_engine = CustomEmbeddingQueryEngine()

results = custom_query_engine.search_by_text_similarity(
    query_text="meeting discussion",
    limit=5
)

for result in results:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Transcript: {result['transcription']}")
    print()
```

## 🎯 **Real-World Use Cases**

### **19. Meeting Transcription System**
```python
from audio_storage_system import AudioStorageSystem
import datetime

class MeetingTranscriptionSystem:
    def __init__(self):
        self.storage = AudioStorageSystem(storage_backend="postgres")
    
    def start_meeting_recording(self, meeting_id, participants):
        """Start recording a meeting."""
        metadata = {
            'meeting_id': meeting_id,
            'participants': ','.join(participants),
            'session_type': 'meeting',
            'start_time': datetime.datetime.now().isoformat(),
            'status': 'recording'
        }
        
        return self.storage.record_and_store_audio(
            duration=3600,  # 1 hour
            whisper_model="medium",  # Higher accuracy for meetings
            generate_embeddings=True,
            metadata=metadata
        )
    
    def get_meeting_transcript(self, meeting_id):
        """Get transcript for a specific meeting."""
        self.storage.cursor.execute("""
            SELECT t.transcription, t.confidence, af.created_at
            FROM audio_files af
            JOIN transcripts t ON af.id = t.audio_file_id
            JOIN metadata m ON af.id = m.audio_file_id
            WHERE m.key = 'meeting_id' AND m.value = %s
            ORDER BY af.created_at
        """, (meeting_id,))
        
        return self.storage.cursor.fetchall()
    
    def search_meeting_content(self, query, meeting_id=None):
        """Search for content in meetings."""
        if meeting_id:
            # Search within specific meeting
            embeddings = self.storage.query_engine.get_embeddings_by_metadata(
                'meeting_id', meeting_id
            )
        else:
            # Search all meetings
            embeddings = self.storage.query_engine.get_all_embeddings('transcript')
        
        # Perform similarity search
        results = self.storage.query_engine.search_by_text_similarity(
            query, limit=10
        )
        
        return results

# Use the meeting system
meeting_system = MeetingTranscriptionSystem()

# Start recording
meeting_id = "meeting_20240101_001"
participants = ["john", "jane", "bob"]
audio_id = meeting_system.start_meeting_recording(meeting_id, participants)

# Get transcript
transcript = meeting_system.get_meeting_transcript(meeting_id)
for entry in transcript:
    print(f"[{entry['created_at']}] {entry['transcription']}")

# Search for specific topics
results = meeting_system.search_meeting_content("project timeline", meeting_id)
for result in results:
    print(f"Found: {result['transcription']}")
```

### **20. Voice Note Taking System**
```python
from audio_storage_system import AudioStorageSystem
import datetime

class VoiceNoteSystem:
    def __init__(self):
        self.storage = AudioStorageSystem(storage_backend="postgres")
    
    def create_voice_note(self, title, category, tags=None):
        """Create a new voice note."""
        metadata = {
            'title': title,
            'category': category,
            'tags': tags or '',
            'type': 'voice_note',
            'created_at': datetime.datetime.now().isoformat()
        }
        
        print(f"Recording voice note: {title}")
        print("Speak now...")
        
        audio_id = self.storage.record_and_store_audio(
            duration=60,  # 1 minute default
            whisper_model="base",
            generate_embeddings=True,
            metadata=metadata
        )
        
        return audio_id
    
    def search_notes(self, query, category=None):
        """Search voice notes by content or category."""
        if category:
            embeddings = self.storage.query_engine.get_embeddings_by_metadata(
                'category', category
            )
        else:
            embeddings = self.storage.query_engine.get_all_embeddings('transcript')
        
        results = self.storage.query_engine.search_by_text_similarity(
            query, limit=10
        )
        
        return results
    
    def get_notes_by_category(self, category):
        """Get all notes in a specific category."""
        self.storage.cursor.execute("""
            SELECT af.id, t.transcription, m.value as title, af.created_at
            FROM audio_files af
            JOIN transcripts t ON af.id = t.audio_file_id
            JOIN metadata m ON af.id = m.audio_file_id
            WHERE m.key = 'category' AND m.value = %s
            ORDER BY af.created_at DESC
        """, (category,))
        
        return self.storage.cursor.fetchall()

# Use the voice note system
note_system = VoiceNoteSystem()

# Create some voice notes
note_system.create_voice_note("Shopping List", "personal", "shopping,errands")
note_system.create_voice_note("Project Ideas", "work", "ideas,projects")
note_system.create_voice_note("Meeting Notes", "work", "meeting,notes")

# Search for notes
results = note_system.search_notes("shopping", category="personal")
for result in results:
    print(f"Found note: {result['transcription']}")

# Get all work notes
work_notes = note_system.get_notes_by_category("work")
for note in work_notes:
    print(f"Work note: {note['title']} - {note['transcription']}")
```

---

These examples demonstrate the flexibility and power of the WhisperPOC system. You can adapt and combine these patterns to build sophisticated audio processing applications tailored to your specific needs. 