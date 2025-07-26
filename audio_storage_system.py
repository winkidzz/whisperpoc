#!/usr/bin/env python3
"""
Audio Storage System with Transcripts and Embeddings
Supports multiple storage backends and efficient retrieval
"""

import whisper
import sounddevice as sd
import numpy as np
import json
import os
import sqlite3
import hashlib
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import requests
import base64
from pathlib import Path

class AudioStorageSystem:
    def __init__(self, storage_backend="sqlite", storage_path="./audio_storage", 
                 postgres_config=None):
        """
        Initialize audio storage system
        
        Args:
            storage_backend: 'sqlite', 'json', 'postgres', 'vector_db'
            storage_path: Path for storage files
            postgres_config: Dict with PostgreSQL connection details
        """
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # PostgreSQL configuration
        self.postgres_config = postgres_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'audio_storage_db',
            'user': os.getenv('USER', 'postgres'),
            'password': ''
        }
        
        # Initialize Whisper for transcription
        self.whisper_model = None  # Lazy loading
        
        # Initialize storage
        self.init_storage()
    
    def init_storage(self):
        """Initialize storage backend"""
        if self.storage_backend == "sqlite":
            self.init_sqlite()
        elif self.storage_backend == "postgres":
            self.init_postgres()
        elif self.storage_backend == "json":
            self.init_json_storage()
        elif self.storage_backend == "vector_db":
            self.init_vector_db()
        else:
            raise ValueError(f"Unsupported storage backend: {self.storage_backend}")
    
    def init_sqlite(self):
        """Initialize SQLite database"""
        db_path = self.storage_path / "audio_storage.db"
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                duration REAL,
                sample_rate INTEGER,
                channels INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                whisper_model TEXT,
                transcription TEXT,
                confidence REAL,
                language TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id)
            );
            
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                transcript_id INTEGER,
                embedding_type TEXT,  -- 'transcript', 'audio', 'combined'
                embedding_data BLOB,
                embedding_dimension INTEGER,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id),
                FOREIGN KEY (transcript_id) REFERENCES transcripts (id)
            );
            
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id INTEGER,
                key TEXT NOT NULL,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_audio_files_hash ON audio_files(file_hash);
            CREATE INDEX IF NOT EXISTS idx_transcripts_audio_id ON transcripts(audio_file_id);
            CREATE INDEX IF NOT EXISTS idx_embeddings_audio_id ON embeddings(audio_file_id);
            CREATE INDEX IF NOT EXISTS idx_metadata_audio_id ON metadata(audio_file_id);
        """)
        self.conn.commit()
    
    def init_postgres(self):
        """Initialize PostgreSQL database"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # Connect to PostgreSQL
            self.conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            # Use RealDictCursor for dictionary-like access
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Create tables
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id SERIAL PRIMARY KEY,
                    file_hash VARCHAR(255) UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size BIGINT,
                    duration REAL,
                    sample_rate INTEGER,
                    channels INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id SERIAL PRIMARY KEY,
                    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
                    whisper_model VARCHAR(50),
                    transcription TEXT,
                    confidence REAL,
                    language VARCHAR(10),
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
                    transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
                    embedding_type VARCHAR(50),  -- 'transcript', 'audio', 'combined'
                    embedding_data BYTEA,
                    embedding_dimension INTEGER,
                    model_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id SERIAL PRIMARY KEY,
                    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
                    key VARCHAR(255) NOT NULL,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_files_hash ON audio_files(file_hash)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_audio_id ON transcripts(audio_file_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_audio_id ON embeddings(audio_file_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_audio_id ON metadata(audio_file_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_audio_files_created_at ON audio_files(created_at)")
            
            self.conn.commit()
            print("✅ PostgreSQL database initialized successfully!")
            
        except ImportError:
            print("❌ psycopg2 not installed. Please install it with: pip install psycopg2-binary")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize PostgreSQL: {e}")
            raise
    
    def init_json_storage(self):
        """Initialize JSON-based storage"""
        self.audio_index_file = self.storage_path / "audio_index.json"
        self.transcripts_file = self.storage_path / "transcripts.json"
        self.embeddings_file = self.storage_path / "embeddings.json"
        self.metadata_file = self.storage_path / "metadata.json"
        
        # Initialize files if they don't exist
        for file_path in [self.audio_index_file, self.transcripts_file, 
                         self.embeddings_file, self.metadata_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump({}, f)
    
    def init_vector_db(self):
        """Initialize vector database (placeholder for ChromaDB, Pinecone, etc.)"""
        print("🔧 Vector database initialization - implement with your preferred vector DB")
        print("💡 Options: ChromaDB, Pinecone, Weaviate, Qdrant")
        # This is a placeholder - implement with your preferred vector database
        pass
    
    def get_whisper_model(self, model_name="base"):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            print(f"🤖 Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            print("✅ Whisper model loaded!")
        return self.whisper_model
    
    def record_and_store_audio(self, duration=10, whisper_model="base", 
                             generate_embeddings=True, metadata=None):
        """Record audio and store with transcript and embeddings"""
        print(f"🎤 Recording {duration} seconds...")
        
        # Record audio
        sample_rate = 16000
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        # Show recording progress
        for i in range(duration):
            print(f"   Recording... {i+1}/{duration}")
            time.sleep(1)
        
        sd.wait()
        audio = audio.flatten()
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.wav"
        audio_path = self.storage_path / "audio_files" / audio_filename
        audio_path.parent.mkdir(exist_ok=True)
        
        import scipy.io.wavfile as wav
        wav.write(str(audio_path), sample_rate, audio)
        
        # Store in database
        audio_id = self.store_audio_file(str(audio_path), audio, metadata)
        
        # Generate transcript
        transcript_id = self.generate_transcript(audio_id, audio, whisper_model)
        
        # Generate embeddings
        if generate_embeddings:
            self.generate_embeddings(audio_id, transcript_id, audio)
        
        return audio_id
    
    def store_audio_file(self, file_path: str, audio_data: np.ndarray, 
                        metadata: Optional[Dict] = None) -> int:
        """Store audio file information"""
        # Calculate file hash
        file_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        
        # Get file info
        file_size = len(audio_data.tobytes())
        duration = len(audio_data) / 16000  # Assuming 16kHz sample rate
        
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            
            # Check if file already exists
            cursor.execute("SELECT id FROM audio_files WHERE file_hash = ?", (file_hash,))
            existing = cursor.fetchone()
            
            if existing:
                print(f"⚠️  Audio file already exists with hash: {file_hash}")
                return existing['id']
            
            # Insert audio file
            cursor.execute("""
                INSERT INTO audio_files (file_hash, file_path, file_size, duration, 
                                       sample_rate, channels, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (file_hash, file_path, file_size, duration, 16000, 1))
            
            audio_id = cursor.lastrowid
            
            # Store metadata
            if metadata:
                for key, value in metadata.items():
                    cursor.execute("""
                        INSERT INTO metadata (audio_file_id, key, value)
                        VALUES (?, ?, ?)
                    """, (audio_id, key, str(value)))
            
            self.conn.commit()
            return audio_id
        
        elif self.storage_backend == "postgres":
            # Check if file already exists
            self.cursor.execute("SELECT id FROM audio_files WHERE file_hash = %s", (file_hash,))
            existing = self.cursor.fetchone()
            
            if existing:
                print(f"⚠️  Audio file already exists with hash: {file_hash}")
                return existing['id']
            
            # Insert audio file
            self.cursor.execute("""
                INSERT INTO audio_files (file_hash, file_path, file_size, duration, 
                                       sample_rate, channels, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (file_hash, file_path, file_size, duration, 16000, 1))
            
            audio_id = self.cursor.fetchone()['id']
            
            # Store metadata
            if metadata:
                for key, value in metadata.items():
                    self.cursor.execute("""
                        INSERT INTO metadata (audio_file_id, key, value)
                        VALUES (%s, %s, %s)
                    """, (audio_id, key, str(value)))
            
            self.conn.commit()
            return audio_id
        
        elif self.storage_backend == "json":
            with open(self.audio_index_file, 'r') as f:
                audio_index = json.load(f)
            
            audio_id = str(len(audio_index) + 1)
            audio_index[audio_id] = {
                "file_hash": file_hash,
                "file_path": file_path,
                "file_size": file_size,
                "duration": duration,
                "sample_rate": 16000,
                "channels": 1,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            with open(self.audio_index_file, 'w') as f:
                json.dump(audio_index, f, indent=2)
            
            return audio_id
    
    def generate_transcript(self, audio_id: int, audio_data: np.ndarray, 
                           whisper_model: str = "base") -> int:
        """Generate and store transcript"""
        print("🎯 Generating transcript...")
        
        # Get Whisper model
        model = self.get_whisper_model(whisper_model)
        
        # Transcribe
        start_time = time.time()
        result = model.transcribe(audio_data, language="en", task="transcribe", fp16=False)
        processing_time = time.time() - start_time
        
        transcription = result["text"].strip()
        confidence = result.get("confidence", 0.0)
        language = result.get("language", "en")
        
        print(f"✅ Transcription ({processing_time:.2f}s): '{transcription}'")
        
        # Store transcript
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO transcripts (audio_file_id, whisper_model, transcription, 
                                       confidence, language, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (audio_id, whisper_model, transcription, confidence, language, processing_time))
            
            transcript_id = cursor.lastrowid
            self.conn.commit()
            return transcript_id
        
        elif self.storage_backend == "postgres":
            self.cursor.execute("""
                INSERT INTO transcripts (audio_file_id, whisper_model, transcription, 
                                       confidence, language, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (audio_id, whisper_model, transcription, confidence, language, processing_time))
            
            transcript_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return transcript_id
        
        elif self.storage_backend == "json":
            with open(self.transcripts_file, 'r') as f:
                transcripts = json.load(f)
            
            transcript_id = str(len(transcripts) + 1)
            transcripts[transcript_id] = {
                "audio_file_id": str(audio_id),
                "whisper_model": whisper_model,
                "transcription": transcription,
                "confidence": confidence,
                "language": language,
                "processing_time": processing_time,
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.transcripts_file, 'w') as f:
                json.dump(transcripts, f, indent=2)
            
            return transcript_id
    
    def generate_embeddings(self, audio_id: int, transcript_id: int, 
                           audio_data: np.ndarray, embedding_models=None):
        """Generate and store embeddings"""
        if embedding_models is None:
            embedding_models = ["transcript"]  # Default to transcript embeddings only
        
        print("🧠 Generating embeddings...")
        
        for embedding_type in embedding_models:
            if embedding_type == "transcript":
                # Get transcript text
                transcript_text = self.get_transcript_text(transcript_id)
                if transcript_text:
                    embedding = self.generate_text_embedding(transcript_text)
                    self.store_embedding(audio_id, transcript_id, "transcript", 
                                       embedding, "text-embedding-ada-002")
            
            elif embedding_type == "audio":
                # Generate audio embeddings (placeholder)
                embedding = self.generate_audio_embedding(audio_data)
                self.store_embedding(audio_id, transcript_id, "audio", 
                                   embedding, "audio-embedding-model")
            
            elif embedding_type == "combined":
                # Combine transcript and audio embeddings
                transcript_embedding = self.generate_text_embedding(
                    self.get_transcript_text(transcript_id)
                )
                audio_embedding = self.generate_audio_embedding(audio_data)
                combined_embedding = np.concatenate([transcript_embedding, audio_embedding])
                self.store_embedding(audio_id, transcript_id, "combined", 
                                   combined_embedding, "combined-model")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding (placeholder - implement with your preferred model)"""
        # This is a placeholder - implement with OpenAI, sentence-transformers, etc.
        # For now, return a dummy embedding
        print("🔧 Text embedding generation - implement with your preferred model")
        print("💡 Options: OpenAI text-embedding-ada-002, sentence-transformers, etc.")
        
        # Dummy embedding for demonstration
        return np.random.rand(1536)  # OpenAI embedding dimension
    
    def generate_audio_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate audio embedding (placeholder)"""
        # This is a placeholder - implement with audio embedding models
        print("🔧 Audio embedding generation - implement with audio embedding models")
        
        # Dummy embedding for demonstration
        return np.random.rand(512)  # Audio embedding dimension
    
    def store_embedding(self, audio_id: int, transcript_id: int, embedding_type: str, 
                       embedding: np.ndarray, model_name: str):
        """Store embedding in database"""
        embedding_bytes = pickle.dumps(embedding)
        embedding_dim = len(embedding)
        
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO embeddings (audio_file_id, transcript_id, embedding_type, 
                                      embedding_data, embedding_dimension, model_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (audio_id, transcript_id, embedding_type, embedding_bytes, 
                 embedding_dim, model_name))
            self.conn.commit()
        
        elif self.storage_backend == "postgres":
            self.cursor.execute("""
                INSERT INTO embeddings (audio_file_id, transcript_id, embedding_type, 
                                      embedding_data, embedding_dimension, model_name)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (audio_id, transcript_id, embedding_type, embedding_bytes, 
                 embedding_dim, model_name))
            self.conn.commit()
        
        elif self.storage_backend == "json":
            with open(self.embeddings_file, 'r') as f:
                embeddings = json.load(f)
            
            embedding_id = str(len(embeddings) + 1)
            embeddings[embedding_id] = {
                "audio_file_id": str(audio_id),
                "transcript_id": str(transcript_id),
                "embedding_type": embedding_type,
                "embedding_data": base64.b64encode(embedding_bytes).decode('utf-8'),
                "embedding_dimension": embedding_dim,
                "model_name": model_name,
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.embeddings_file, 'w') as f:
                json.dump(embeddings, f, indent=2)
    
    def get_transcript_text(self, transcript_id: int) -> Optional[str]:
        """Get transcript text by ID"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("SELECT transcription FROM transcripts WHERE id = ?", (transcript_id,))
            result = cursor.fetchone()
            return result['transcription'] if result else None
        
        elif self.storage_backend == "postgres":
            self.cursor.execute("SELECT transcription FROM transcripts WHERE id = %s", (transcript_id,))
            result = self.cursor.fetchone()
            return result['transcription'] if result else None
        
        elif self.storage_backend == "json":
            with open(self.transcripts_file, 'r') as f:
                transcripts = json.load(f)
            
            transcript = transcripts.get(str(transcript_id))
            return transcript['transcription'] if transcript else None
    
    def search_similar_audio(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Search for similar audio based on transcript embeddings"""
        print(f"🔍 Searching for similar audio: '{query_text}'")
        
        # Generate query embedding
        query_embedding = self.generate_text_embedding(query_text)
        
        # Search in database
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT af.file_path, t.transcription, e.embedding_data, e.embedding_dimension
                FROM audio_files af
                JOIN transcripts t ON af.id = t.audio_file_id
                JOIN embeddings e ON af.id = e.audio_file_id
                WHERE e.embedding_type = 'transcript'
                ORDER BY af.created_at DESC
                LIMIT ?
            """, (limit * 2,))  # Get more for similarity calculation
            
            results = []
            for row in cursor.fetchall():
                stored_embedding = pickle.loads(row['embedding_data'])
                similarity = self.calculate_similarity(query_embedding, stored_embedding)
                
                results.append({
                    'file_path': row['file_path'],
                    'transcription': row['transcription'],
                    'similarity': similarity
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
        
        elif self.storage_backend == "postgres":
            self.cursor.execute("""
                SELECT af.file_path, t.transcription, e.embedding_data, e.embedding_dimension
                FROM audio_files af
                JOIN transcripts t ON af.id = t.audio_file_id
                JOIN embeddings e ON af.id = e.audio_file_id
                WHERE e.embedding_type = 'transcript'
                ORDER BY af.created_at DESC
                LIMIT %s
            """, (limit * 2,))  # Get more for similarity calculation
            
            results = []
            for row in self.cursor.fetchall():
                stored_embedding = pickle.loads(row['embedding_data'])
                similarity = self.calculate_similarity(query_embedding, stored_embedding)
                
                results.append({
                    'file_path': row['file_path'],
                    'transcription': row['transcription'],
                    'similarity': similarity
                })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
        
        elif self.storage_backend == "json":
            # Implement JSON-based search
            print("🔧 JSON-based search not implemented yet")
            return []
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_audio_info(self, audio_id: int) -> Optional[Dict]:
        """Get comprehensive audio information"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            
            # Get audio file info
            cursor.execute("""
                SELECT * FROM audio_files WHERE id = ?
            """, (audio_id,))
            audio_file = cursor.fetchone()
            
            if not audio_file:
                return None
            
            # Get transcript
            cursor.execute("""
                SELECT * FROM transcripts WHERE audio_file_id = ?
                ORDER BY created_at DESC LIMIT 1
            """, (audio_id,))
            transcript = cursor.fetchone()
            
            # Get embeddings
            cursor.execute("""
                SELECT embedding_type, model_name, embedding_dimension, created_at
                FROM embeddings WHERE audio_file_id = ?
            """, (audio_id,))
            embeddings = cursor.fetchall()
            
            # Get metadata
            cursor.execute("""
                SELECT key, value FROM metadata WHERE audio_file_id = ?
            """, (audio_id,))
            metadata = dict(cursor.fetchall())
            
            return {
                'audio_file': dict(audio_file),
                'transcript': dict(transcript) if transcript else None,
                'embeddings': [dict(emb) for emb in embeddings],
                'metadata': metadata
            }
        
        elif self.storage_backend == "postgres":
            # Get audio file info
            self.cursor.execute("""
                SELECT * FROM audio_files WHERE id = %s
            """, (audio_id,))
            audio_file = self.cursor.fetchone()
            
            if not audio_file:
                return None
            
            # Get transcript
            self.cursor.execute("""
                SELECT * FROM transcripts WHERE audio_file_id = %s
                ORDER BY created_at DESC LIMIT 1
            """, (audio_id,))
            transcript = self.cursor.fetchone()
            
            # Get embeddings
            self.cursor.execute("""
                SELECT embedding_type, model_name, embedding_dimension, created_at
                FROM embeddings WHERE audio_file_id = %s
            """, (audio_id,))
            embeddings = self.cursor.fetchall()
            
            # Get metadata
            self.cursor.execute("""
                SELECT key, value FROM metadata WHERE audio_file_id = %s
            """, (audio_id,))
            metadata = dict(self.cursor.fetchall())
            
            return {
                'audio_file': dict(audio_file),
                'transcript': dict(transcript) if transcript else None,
                'embeddings': [dict(emb) for emb in embeddings],
                'metadata': metadata
            }
        
        elif self.storage_backend == "json":
            # Implement JSON-based retrieval
            print("🔧 JSON-based retrieval not implemented yet")
            return None
    
    def list_all_audio(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List all stored audio files with pagination support"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT af.*, t.transcription, t.confidence, t.language, 
                       (SELECT COUNT(*) FROM embeddings WHERE audio_file_id = af.id) as embedding_count
                FROM audio_files af
                LEFT JOIN transcripts t ON af.id = t.audio_file_id
                ORDER BY af.created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            return [dict(row) for row in cursor.fetchall()]
        
        elif self.storage_backend == "postgres":
            self.cursor.execute("""
                SELECT af.*, t.transcription, t.confidence, t.language, 
                       (SELECT COUNT(*) FROM embeddings WHERE audio_file_id = af.id) as embedding_count
                FROM audio_files af
                LEFT JOIN transcripts t ON af.id = t.audio_file_id
                ORDER BY af.created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            return [dict(row) for row in self.cursor.fetchall()]
        
        elif self.storage_backend == "json":
            with open(self.audio_index_file, 'r') as f:
                audio_index = json.load(f)
            
            audio_list = list(audio_index.values())
            return audio_list[offset:offset + limit]
    
    def get_audio_count(self) -> int:
        """Get total number of audio files"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM audio_files")
            return cursor.fetchone()['count']
        elif self.storage_backend == "postgres":
            self.cursor.execute("SELECT COUNT(*) as count FROM audio_files")
            return self.cursor.fetchone()['count']
        elif self.storage_backend == "json":
            # Count files in JSON storage
            return len(self.audio_index)
        return 0

    def get_transcript_count(self) -> int:
        """Get total number of transcripts"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM transcripts")
            return cursor.fetchone()['count']
        elif self.storage_backend == "postgres":
            self.cursor.execute("SELECT COUNT(*) as count FROM transcripts")
            return self.cursor.fetchone()['count']
        elif self.storage_backend == "json":
            # Count transcripts in JSON storage
            count = 0
            for audio_info in self.audio_index.values():
                if 'transcript' in audio_info:
                    count += 1
            return count
        return 0

    def get_embedding_count(self) -> int:
        """Get total number of embeddings"""
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM embeddings")
            return cursor.fetchone()['count']
        elif self.storage_backend == "postgres":
            self.cursor.execute("SELECT COUNT(*) as count FROM embeddings")
            return self.cursor.fetchone()['count']
        elif self.storage_backend == "json":
            # Count embeddings in JSON storage
            count = 0
            for audio_info in self.audio_index.values():
                if 'embeddings' in audio_info:
                    count += len(audio_info['embeddings'])
            return count
        return 0

    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old audio files and database entries"""
        print(f"🧹 Cleaning up files older than {days_old} days...")
        
        if self.storage_backend == "sqlite":
            cursor = self.conn.cursor()
            
            # Get old audio files
            cursor.execute("""
                SELECT file_path FROM audio_files 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            old_files = cursor.fetchall()
            
            for row in old_files:
                file_path = row['file_path']
                
                # Delete file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Deleted: {file_path}")
                
                # Delete from database (cascade will handle related records)
                cursor.execute("DELETE FROM audio_files WHERE file_path = ?", (file_path,))
            
            self.conn.commit()
            print(f"✅ Cleaned up {len(old_files)} old files")
        
        elif self.storage_backend == "postgres":
            # Get old audio files
            self.cursor.execute("""
                SELECT file_path FROM audio_files 
                WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '{} days'
            """.format(days_old))
            
            old_files = self.cursor.fetchall()
            
            for row in old_files:
                file_path = row['file_path']
                
                # Delete file if it exists
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Deleted: {file_path}")
                
                # Delete from database (cascade will handle related records)
                self.cursor.execute("DELETE FROM audio_files WHERE file_path = %s", (file_path,))
            
            self.conn.commit()
            print(f"✅ Cleaned up {len(old_files)} old files")
        
        elif self.storage_backend == "json":
            # Implement JSON-based cleanup
            print("🔧 JSON-based cleanup not implemented yet")

def main():
    """Main function for testing the audio storage system"""
    print("🗄️  Audio Storage System with PostgreSQL")
    print("=" * 60)
    
    # Initialize storage system with PostgreSQL
    storage = AudioStorageSystem(storage_backend="postgres")
    
    # Record and store audio
    print("\n1. Recording and storing audio...")
    audio_id = storage.record_and_store_audio(
        duration=5,
        whisper_model="base",
        generate_embeddings=True,
        metadata={"source": "microphone", "user": "test", "session": "postgres_test"}
    )
    
    print(f"✅ Audio stored with ID: {audio_id}")
    
    # Get audio info
    print("\n2. Retrieving audio information...")
    audio_info = storage.get_audio_info(audio_id)
    if audio_info:
        print(f"📁 File: {audio_info['audio_file']['file_path']}")
        print(f"🎯 Transcript: {audio_info['transcript']['transcription']}")
        print(f"🧠 Embeddings: {len(audio_info['embeddings'])} types")
        print(f"🏷️  Metadata: {audio_info['metadata']}")
    
    # Search similar audio
    print("\n3. Searching for similar audio...")
    results = storage.search_similar_audio("test audio recording", limit=5)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['transcription'][:50]}... (similarity: {result['similarity']:.3f})")
    
    # List all audio
    print("\n4. Listing all stored audio...")
    all_audio = storage.list_all_audio(limit=10)
    print(f"📊 Total stored audio files: {len(all_audio)}")
    
    # Close database connection
    if hasattr(storage, 'conn'):
        storage.conn.close()
        print("\n🔒 Database connection closed")

if __name__ == "__main__":
    main() 