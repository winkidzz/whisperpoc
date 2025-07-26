# Vector Embedding Query Guide for PostgreSQL

## 🧠 **Overview**

This guide shows you how to query and analyze vector embeddings stored in PostgreSQL for your audio storage system. Embeddings are stored as binary BLOB data and can be queried using various methods.

## 📊 **Current Embedding Status**

Based on your database:
- **Total Embeddings:** 3 transcript embeddings
- **Dimension:** 1536 (OpenAI text-embedding-ada-002 format)
- **Storage Size:** 0.04 MB
- **Model:** text-embedding-ada-002

## 🔍 **Query Methods**

### **1. Direct PostgreSQL Queries**

#### **View All Embeddings**
```sql
-- Basic embedding information
SELECT 
    e.id,
    e.embedding_type,
    e.model_name,
    e.embedding_dimension,
    e.created_at,
    t.transcription,
    af.file_path
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
JOIN audio_files af ON e.audio_file_id = af.id
ORDER BY e.created_at DESC;
```

#### **Get Embedding Statistics**
```sql
-- Count embeddings by type and model
SELECT 
    embedding_type,
    model_name,
    COUNT(*) as count,
    AVG(embedding_dimension) as avg_dimension,
    MIN(embedding_dimension) as min_dimension,
    MAX(embedding_dimension) as max_dimension
FROM embeddings
GROUP BY embedding_type, model_name
ORDER BY embedding_type, model_name;
```

#### **Filter by Metadata**
```sql
-- Find embeddings for specific user
SELECT 
    e.id,
    e.embedding_type,
    t.transcription,
    af.file_path,
    m.value as user
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
JOIN audio_files af ON e.audio_file_id = af.id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'user' AND m.value = 's1'
ORDER BY e.created_at DESC;
```

### **2. Python Query Methods**

#### **Initialize Query Engine**
```python
from embedding_queries import EmbeddingQueryEngine

# Initialize the query engine
query_engine = EmbeddingQueryEngine(storage_backend="postgres")
```

#### **Get All Embeddings**
```python
# Get all transcript embeddings
embeddings = query_engine.get_all_embeddings(embedding_type="transcript")

# Get all audio embeddings
embeddings = query_engine.get_all_embeddings(embedding_type="audio")

# Get all combined embeddings
embeddings = query_engine.get_all_embeddings(embedding_type="combined")
```

#### **Analyze Specific Embedding**
```python
# Analyze embedding by ID
embedding_data, embedding_array = query_engine.analyze_embedding(embedding_id=1)

# This returns:
# - embedding_data: Database record with metadata
# - embedding_array: Numpy array of the actual embedding vector
```

#### **Similarity Search**
```python
# Find similar embeddings using cosine similarity
import numpy as np

# Create a query embedding (in real use, this would come from an embedding model)
query_embedding = np.random.rand(1536)

# Find similar embeddings
similar_embeddings = query_engine.find_similar_embeddings(
    query_embedding=query_embedding,
    limit=5,
    embedding_type="transcript"
)
```

#### **Text-based Search**
```python
# Search for similar audio based on text
similar_embeddings = query_engine.search_by_text_similarity(
    query_text="Hulk news",
    limit=3
)
```

#### **Metadata-based Search**
```python
# Find embeddings by user
user_embeddings = query_engine.get_embeddings_by_metadata(
    metadata_key="user",
    metadata_value="s1"
)

# Find embeddings by session
session_embeddings = query_engine.get_embeddings_by_metadata(
    metadata_key="session",
    metadata_value="sn1"
)
```

### **3. Advanced Query Techniques**

#### **Export Embeddings**
```python
# Export all embeddings to numpy file
export_file = query_engine.export_embeddings_to_numpy("my_embeddings.npz")

# Load embeddings back
embeddings, metadata = query_engine.load_embeddings_from_numpy("my_embeddings.npz")
```

#### **Batch Similarity Analysis**
```python
# Compare all embeddings with each other
embeddings = query_engine.get_all_embeddings()
embedding_arrays = [pickle.loads(emb['embedding_data']) for emb in embeddings]

# Create similarity matrix
n = len(embedding_arrays)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i][j] = query_engine.calculate_cosine_similarity(
            embedding_arrays[i], 
            embedding_arrays[j]
        )

print("Similarity Matrix:")
print(similarity_matrix)
```

#### **Embedding Clustering**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Get all embeddings
embeddings = query_engine.get_all_embeddings()
embedding_arrays = [pickle.loads(emb['embedding_data']) for emb in embeddings]
embedding_matrix = np.array(embedding_arrays)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embedding_matrix)

# Cluster embeddings
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embedding_matrix)

print("Embedding Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Embedding {i+1}: Cluster {cluster}")
```

## 🔧 **Raw SQL Queries for Embeddings**

### **Basic Embedding Queries**
```sql
-- View embedding table structure
\d embeddings

-- Count total embeddings
SELECT COUNT(*) FROM embeddings;

-- Get embedding types
SELECT DISTINCT embedding_type FROM embeddings;

-- Get model names
SELECT DISTINCT model_name FROM embeddings;

-- Get dimension range
SELECT 
    MIN(embedding_dimension) as min_dim,
    MAX(embedding_dimension) as max_dim,
    AVG(embedding_dimension) as avg_dim
FROM embeddings;
```

### **Complex Joins**
```sql
-- Get embeddings with full context
SELECT 
    e.id as embedding_id,
    e.embedding_type,
    e.model_name,
    e.embedding_dimension,
    af.id as audio_file_id,
    af.file_path,
    af.duration,
    t.transcription,
    t.confidence,
    string_agg(m.key || ':' || m.value, ', ') as metadata
FROM embeddings e
JOIN audio_files af ON e.audio_file_id = af.id
JOIN transcripts t ON e.transcript_id = t.id
LEFT JOIN metadata m ON af.id = m.audio_file_id
GROUP BY e.id, e.embedding_type, e.model_name, e.embedding_dimension,
         af.id, af.file_path, af.duration, t.transcription, t.confidence
ORDER BY e.created_at DESC;
```

### **Storage Analysis**
```sql
-- Embedding storage size analysis
SELECT 
    embedding_type,
    model_name,
    COUNT(*) as count,
    SUM(LENGTH(embedding_data)) as total_size_bytes,
    AVG(LENGTH(embedding_data)) as avg_size_bytes
FROM embeddings
GROUP BY embedding_type, model_name
ORDER BY total_size_bytes DESC;
```

## 🚀 **Performance Optimization**

### **Indexing for Embedding Queries**
```sql
-- Create indexes for better performance
CREATE INDEX idx_embeddings_type ON embeddings(embedding_type);
CREATE INDEX idx_embeddings_model ON embeddings(model_name);
CREATE INDEX idx_embeddings_audio_id ON embeddings(audio_file_id);
CREATE INDEX idx_embeddings_transcript_id ON embeddings(transcript_id);
CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);
```

### **Partitioning for Large Datasets**
```sql
-- Partition embeddings by type (for very large datasets)
CREATE TABLE embeddings_partitioned (
    LIKE embeddings INCLUDING ALL
) PARTITION BY LIST (embedding_type);

CREATE TABLE embeddings_transcript PARTITION OF embeddings_partitioned
    FOR VALUES IN ('transcript');
CREATE TABLE embeddings_audio PARTITION OF embeddings_partitioned
    FOR VALUES IN ('audio');
CREATE TABLE embeddings_combined PARTITION OF embeddings_partitioned
    FOR VALUES IN ('combined');
```

## 🔮 **Advanced Features**

### **Vector Extensions (pgvector)**
For production use, consider installing the pgvector extension:

```sql
-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector column
ALTER TABLE embeddings ADD COLUMN embedding_vector vector(1536);

-- Convert BLOB to vector (example)
UPDATE embeddings 
SET embedding_vector = embedding_data::vector 
WHERE embedding_type = 'transcript';

-- Vector similarity search
SELECT 
    e.id,
    t.transcription,
    e.embedding_vector <=> '[0.1, 0.2, ...]'::vector as distance
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
ORDER BY distance
LIMIT 5;
```

### **Real-time Embedding Generation**
```python
# Example with real embedding model
from sentence_transformers import SentenceTransformer

def generate_real_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)

# Update embedding generation in audio_storage_system.py
def generate_text_embedding(self, text: str) -> np.ndarray:
    return generate_real_embeddings(text)
```

## 📊 **Query Examples for Your Data**

Based on your current database, here are some useful queries:

### **View Your Embeddings**
```bash
psql -d audio_storage_db -c "
SELECT 
    e.id,
    e.embedding_type,
    t.transcription,
    af.file_path,
    e.embedding_dimension
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
JOIN audio_files af ON e.audio_file_id = af.id
ORDER BY e.id;
"
```

### **Find Similar Content**
```python
# Run the embedding query engine
python embedding_queries.py
```

### **Export Your Embeddings**
```python
from embedding_queries import EmbeddingQueryEngine

query_engine = EmbeddingQueryEngine()
export_file = query_engine.export_embeddings_to_numpy("my_audio_embeddings.npz")
```

## 🎯 **Best Practices**

1. **Use Indexes** - Always index embedding-related columns
2. **Batch Operations** - Process multiple embeddings together
3. **Vector Extensions** - Use pgvector for production similarity search
4. **Regular Maintenance** - Analyze tables and update statistics
5. **Backup Strategy** - Regularly backup embedding data
6. **Monitoring** - Track embedding storage growth and query performance

Your PostgreSQL embedding storage is ready for advanced vector operations! 🚀 