#!/usr/bin/env python3
"""
Vector Embedding Queries for PostgreSQL Audio Storage
Demonstrates how to query and analyze embeddings stored in PostgreSQL
"""

import numpy as np
import pickle
import time
from audio_storage_system import AudioStorageSystem

class EmbeddingQueryEngine:
    def __init__(self, storage_backend="postgres"):
        """Initialize embedding query engine"""
        self.storage = AudioStorageSystem(storage_backend=storage_backend)
    
    def get_all_embeddings(self, embedding_type="transcript"):
        """Retrieve all embeddings of a specific type"""
        print(f"🔍 Retrieving all {embedding_type} embeddings...")
        
        self.storage.cursor.execute("""
            SELECT 
                e.id,
                e.audio_file_id,
                e.transcript_id,
                e.embedding_data,
                e.embedding_dimension,
                e.model_name,
                e.created_at,
                t.transcription,
                af.file_path
            FROM embeddings e
            JOIN transcripts t ON e.transcript_id = t.id
            JOIN audio_files af ON e.audio_file_id = af.id
            WHERE e.embedding_type = %s
            ORDER BY e.created_at DESC
        """, (embedding_type,))
        
        embeddings = self.storage.cursor.fetchall()
        print(f"✅ Found {len(embeddings)} {embedding_type} embeddings")
        
        return embeddings
    
    def analyze_embedding(self, embedding_id):
        """Analyze a specific embedding"""
        print(f"🔍 Analyzing embedding ID: {embedding_id}")
        
        self.storage.cursor.execute("""
            SELECT 
                e.*,
                t.transcription,
                af.file_path,
                af.duration
            FROM embeddings e
            JOIN transcripts t ON e.transcript_id = t.id
            JOIN audio_files af ON e.audio_file_id = af.id
            WHERE e.id = %s
        """, (embedding_id,))
        
        embedding_data = self.storage.cursor.fetchone()
        if not embedding_data:
            print(f"❌ Embedding ID {embedding_id} not found")
            return None
        
        # Decode the embedding
        embedding_array = pickle.loads(embedding_data['embedding_data'])
        
        print(f"📊 Embedding Analysis:")
        print(f"   🆔 ID: {embedding_data['id']}")
        print(f"   📁 Audio File: {embedding_data['file_path']}")
        print(f"   🎯 Transcript: {embedding_data['transcription']}")
        print(f"   📏 Dimension: {embedding_data['embedding_dimension']}")
        print(f"   🤖 Model: {embedding_data['model_name']}")
        print(f"   ⏱️  Duration: {embedding_data['duration']:.1f}s")
        print(f"   📊 Embedding Stats:")
        print(f"      - Shape: {embedding_array.shape}")
        print(f"      - Mean: {np.mean(embedding_array):.4f}")
        print(f"      - Std: {np.std(embedding_array):.4f}")
        print(f"      - Min: {np.min(embedding_array):.4f}")
        print(f"      - Max: {np.max(embedding_array):.4f}")
        print(f"      - Norm: {np.linalg.norm(embedding_array):.4f}")
        
        return embedding_data, embedding_array
    
    def find_similar_embeddings(self, query_embedding, limit=5, embedding_type="transcript"):
        """Find similar embeddings using cosine similarity"""
        print(f"🔍 Finding similar {embedding_type} embeddings...")
        
        # Get all embeddings of the specified type
        embeddings = self.get_all_embeddings(embedding_type)
        
        similarities = []
        for emb in embeddings:
            stored_embedding = pickle.loads(emb['embedding_data'])
            similarity = self.calculate_cosine_similarity(query_embedding, stored_embedding)
            
            similarities.append({
                'embedding_id': emb['id'],
                'audio_file_id': emb['audio_file_id'],
                'transcription': emb['transcription'],
                'file_path': emb['file_path'],
                'similarity': similarity,
                'model_name': emb['model_name']
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"📊 Top {limit} similar embeddings:")
        for i, sim in enumerate(similarities[:limit], 1):
            print(f"   {i}. Similarity: {sim['similarity']:.4f}")
            print(f"      🆔 Embedding ID: {sim['embedding_id']}")
            print(f"      🎯 Transcript: {sim['transcription'][:50]}...")
            print(f"      📁 File: {sim['file_path']}")
            print(f"      🤖 Model: {sim['model_name']}")
            print()
        
        return similarities[:limit]
    
    def calculate_cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_embedding_statistics(self):
        """Get statistics about all embeddings in the database"""
        print("📊 Embedding Database Statistics")
        print("=" * 50)
        
        # Count embeddings by type
        self.storage.cursor.execute("""
            SELECT 
                embedding_type,
                model_name,
                COUNT(*) as count,
                AVG(embedding_dimension) as avg_dimension,
                MIN(embedding_dimension) as min_dimension,
                MAX(embedding_dimension) as max_dimension
            FROM embeddings
            GROUP BY embedding_type, model_name
            ORDER BY embedding_type, model_name
        """)
        
        stats = self.storage.cursor.fetchall()
        
        print("📈 Embedding Types:")
        for stat in stats:
            print(f"   🏷️  Type: {stat['embedding_type']}")
            print(f"      🤖 Model: {stat['model_name']}")
            print(f"      📊 Count: {stat['count']}")
            print(f"      📏 Dimensions: {stat['min_dimension']} - {stat['max_dimension']} (avg: {stat['avg_dimension']:.1f})")
            print()
        
        # Total storage size
        self.storage.cursor.execute("""
            SELECT 
                COUNT(*) as total_embeddings,
                SUM(LENGTH(embedding_data)) as total_size_bytes
            FROM embeddings
        """)
        
        total_stats = self.storage.cursor.fetchone()
        print(f"💾 Total Storage:")
        print(f"   📊 Total embeddings: {total_stats['total_embeddings']}")
        print(f"   💾 Total size: {total_stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
        
        return stats
    
    def search_by_text_similarity(self, query_text, limit=5):
        """Search for similar audio based on text similarity"""
        print(f"🔍 Searching for audio similar to: '{query_text}'")
        
        # Generate a dummy embedding for the query text
        # In a real implementation, you would use an actual embedding model
        query_embedding = np.random.rand(1536)  # OpenAI embedding dimension
        
        # Find similar embeddings
        similar_embeddings = self.find_similar_embeddings(query_embedding, limit, "transcript")
        
        return similar_embeddings
    
    def get_embeddings_by_metadata(self, metadata_key, metadata_value):
        """Get embeddings filtered by metadata"""
        print(f"🔍 Finding embeddings with {metadata_key} = {metadata_value}")
        
        self.storage.cursor.execute("""
            SELECT 
                e.id,
                e.embedding_type,
                e.model_name,
                e.embedding_dimension,
                t.transcription,
                af.file_path,
                m.value as metadata_value
            FROM embeddings e
            JOIN transcripts t ON e.transcript_id = t.id
            JOIN audio_files af ON e.audio_file_id = af.id
            JOIN metadata m ON af.id = m.audio_file_id
            WHERE m.key = %s AND m.value = %s
            ORDER BY e.created_at DESC
        """, (metadata_key, metadata_value))
        
        embeddings = self.storage.cursor.fetchall()
        
        print(f"✅ Found {len(embeddings)} embeddings matching criteria")
        for emb in embeddings:
            print(f"   🆔 ID: {emb['id']}")
            print(f"   🏷️  Type: {emb['embedding_type']}")
            print(f"   🎯 Transcript: {emb['transcription'][:50]}...")
            print(f"   📁 File: {emb['file_path']}")
            print()
        
        return embeddings
    
    def export_embeddings_to_numpy(self, filename="embeddings_export.npz"):
        """Export all embeddings to a numpy file"""
        print(f"📤 Exporting embeddings to {filename}...")
        
        embeddings = self.get_all_embeddings()
        
        if not embeddings:
            print("❌ No embeddings found to export")
            return
        
        # Prepare data for export
        embedding_arrays = []
        metadata = []
        
        for emb in embeddings:
            embedding_array = pickle.loads(emb['embedding_data'])
            embedding_arrays.append(embedding_array)
            
            metadata.append({
                'id': emb['id'],
                'audio_file_id': emb['audio_file_id'],
                'transcript_id': emb['transcript_id'],
                'transcription': emb['transcription'],
                'file_path': emb['file_path'],
                'model_name': emb['model_name'],
                'dimension': emb['embedding_dimension'],
                'created_at': str(emb['created_at'])
            })
        
        # Convert to numpy arrays
        embedding_matrix = np.array(embedding_arrays)
        
        # Save to file
        np.savez_compressed(
            filename,
            embeddings=embedding_matrix,
            metadata=metadata
        )
        
        print(f"✅ Exported {len(embedding_arrays)} embeddings")
        print(f"   📊 Matrix shape: {embedding_matrix.shape}")
        print(f"   💾 File size: {embedding_matrix.nbytes / 1024 / 1024:.2f} MB")
        
        return filename
    
    def load_embeddings_from_numpy(self, filename="embeddings_export.npz"):
        """Load embeddings from a numpy file"""
        print(f"📥 Loading embeddings from {filename}...")
        
        try:
            data = np.load(filename, allow_pickle=True)
            embeddings = data['embeddings']
            metadata = data['metadata']
            
            print(f"✅ Loaded {len(embeddings)} embeddings")
            print(f"   📊 Matrix shape: {embeddings.shape}")
            
            return embeddings, metadata
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return None, None

def main():
    """Main function to demonstrate embedding queries"""
    print("🧠 Vector Embedding Query Engine")
    print("=" * 60)
    
    # Initialize query engine
    query_engine = EmbeddingQueryEngine()
    
    try:
        # 1. Get embedding statistics
        print("\n1. 📊 Embedding Statistics")
        print("-" * 40)
        query_engine.get_embedding_statistics()
        
        # 2. Analyze a specific embedding
        print("\n2. 🔍 Embedding Analysis")
        print("-" * 40)
        # Get the first embedding ID
        query_engine.storage.cursor.execute("SELECT id FROM embeddings LIMIT 1")
        first_embedding = query_engine.storage.cursor.fetchone()
        
        if first_embedding:
            query_engine.analyze_embedding(first_embedding['id'])
        
        # 3. Search by text similarity
        print("\n3. 🔍 Text Similarity Search")
        print("-" * 40)
        query_engine.search_by_text_similarity("Hulk news", limit=3)
        
        # 4. Search by metadata
        print("\n4. 🔍 Metadata-based Search")
        print("-" * 40)
        query_engine.get_embeddings_by_metadata("user", "s1")
        
        # 5. Export embeddings
        print("\n5. 📤 Export Embeddings")
        print("-" * 40)
        export_file = query_engine.export_embeddings_to_numpy()
        
        # 6. Load embeddings back
        print("\n6. 📥 Load Embeddings")
        print("-" * 40)
        if export_file:
            embeddings, metadata = query_engine.load_embeddings_from_numpy(export_file)
        
        # 7. Advanced similarity search
        print("\n7. 🔍 Advanced Similarity Search")
        print("-" * 40)
        
        # Create a sample query embedding
        query_embedding = np.random.rand(1536)
        print("🔍 Finding similar embeddings to random query vector...")
        similar = query_engine.find_similar_embeddings(query_embedding, limit=3)
        
        print("\n🎉 Embedding query demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Close database connection
        query_engine.storage.conn.close()
        print("🔒 Database connection closed")

if __name__ == "__main__":
    main() 