#!/usr/bin/env python3
"""
Similarity Search Example

This script demonstrates advanced similarity search capabilities for finding
similar audio content based on transcript embeddings and metadata.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_queries import EmbeddingQueryEngine
from audio_storage_system import AudioStorageSystem
import numpy as np

def basic_similarity_search():
    """Demonstrate basic similarity search functionality."""
    
    print("🔍 Basic Similarity Search")
    print("=" * 50)
    
    query_engine = EmbeddingQueryEngine()
    
    # Sample search queries
    search_queries = [
        "meeting discussion",
        "project update",
        "technical problem",
        "customer feedback",
        "team collaboration"
    ]
    
    for query in search_queries:
        print(f"\n🔎 Searching for: '{query}'")
        
        try:
            results = query_engine.search_by_text_similarity(
                query_text=query,
                limit=5
            )
            
            if results:
                print(f"   Found {len(results)} similar recordings:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Similarity: {result['similarity']:.3f}")
                    print(f"      Transcript: '{result['transcription']}'")
                    print(f"      File: {result['file_path']}")
                    print()
            else:
                print("   No similar recordings found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

def metadata_filtered_search():
    """Demonstrate similarity search with metadata filtering."""
    
    print("\n🏷️  Metadata-Filtered Similarity Search")
    print("=" * 50)
    
    query_engine = EmbeddingQueryEngine()
    
    # Search queries with metadata filters
    search_scenarios = [
        {
            'query': 'meeting discussion',
            'metadata_key': 'user',
            'metadata_value': 'john',
            'description': 'Find meeting discussions by user John'
        },
        {
            'query': 'project update',
            'metadata_key': 'priority',
            'metadata_value': 'high',
            'description': 'Find high-priority project updates'
        },
        {
            'query': 'technical problem',
            'metadata_key': 'session',
            'metadata_value': 'team_meeting',
            'description': 'Find technical problems from team meetings'
        }
    ]
    
    for scenario in search_scenarios:
        print(f"\n🔎 {scenario['description']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Filter: {scenario['metadata_key']} = '{scenario['metadata_value']}'")
        
        try:
            # Get embeddings filtered by metadata
            filtered_embeddings = query_engine.get_embeddings_by_metadata(
                metadata_key=scenario['metadata_key'],
                metadata_value=scenario['metadata_value']
            )
            
            if filtered_embeddings:
                print(f"   Found {len(filtered_embeddings)} recordings with filter:")
                for embedding in filtered_embeddings[:3]:  # Show first 3
                    print(f"      - {embedding['transcription']}")
                
                # Perform similarity search on filtered results
                results = query_engine.search_by_text_similarity(
                    query_text=scenario['query'],
                    limit=3
                )
                
                if results:
                    print(f"   Top similar results:")
                    for i, result in enumerate(results, 1):
                        print(f"      {i}. Similarity: {result['similarity']:.3f}")
                        print(f"         '{result['transcription']}'")
                else:
                    print("   No similar results found")
            else:
                print("   No recordings found with specified metadata filter")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

def advanced_similarity_analysis():
    """Demonstrate advanced similarity analysis features."""
    
    print("\n📊 Advanced Similarity Analysis")
    print("=" * 50)
    
    query_engine = EmbeddingQueryEngine()
    
    # Get embedding statistics
    print("📈 Embedding Statistics:")
    stats = query_engine.get_embedding_statistics()
    print(f"   Total embeddings: {stats['total_embeddings']}")
    print(f"   Embedding types: {stats['embedding_types']}")
    print(f"   Average dimension: {stats['avg_dimension']:.1f}")
    print(f"   Total storage: {stats['total_storage_mb']:.2f} MB")
    
    # Analyze specific embeddings
    print(f"\n🔬 Embedding Analysis:")
    all_embeddings = query_engine.get_all_embeddings('transcript')
    
    if all_embeddings:
        # Analyze first few embeddings
        for i, embedding_data in enumerate(all_embeddings[:3], 1):
            print(f"\n   Embedding {i}:")
            analysis = query_engine.analyze_embedding(embedding_data['id'])
            if analysis:
                embedding_array = analysis[1]
                print(f"      Shape: {embedding_array.shape}")
                print(f"      Mean: {np.mean(embedding_array):.4f}")
                print(f"      Std: {np.std(embedding_array):.4f}")
                print(f"      Min: {np.min(embedding_array):.4f}")
                print(f"      Max: {np.max(embedding_array):.4f}")
                print(f"      Norm: {np.linalg.norm(embedding_array):.4f}")

def similarity_clustering():
    """Demonstrate similarity-based clustering of audio content."""
    
    print("\n🎯 Similarity Clustering")
    print("=" * 50)
    
    query_engine = EmbeddingQueryEngine()
    
    # Get all embeddings
    all_embeddings = query_engine.get_all_embeddings('transcript')
    
    if len(all_embeddings) < 2:
        print("   Need at least 2 embeddings for clustering")
        return
    
    print(f"   Analyzing {len(all_embeddings)} embeddings for clustering...")
    
    # Calculate similarity matrix
    similarities = []
    for i, emb1 in enumerate(all_embeddings):
        for j, emb2 in enumerate(all_embeddings[i+1:], i+1):
            similarity = query_engine.calculate_cosine_similarity(
                query_engine.get_embedding_array(emb1['id']),
                query_engine.get_embedding_array(emb2['id'])
            )
            similarities.append({
                'emb1_id': emb1['id'],
                'emb2_id': emb2['id'],
                'similarity': similarity,
                'transcript1': emb1['transcription'],
                'transcript2': emb2['transcription']
            })
    
    # Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Show most similar pairs
    print(f"\n   Top Similar Pairs:")
    for i, pair in enumerate(similarities[:5], 1):
        print(f"   {i}. Similarity: {pair['similarity']:.3f}")
        print(f"      '{pair['transcript1']}'")
        print(f"      '{pair['transcript2']}'")
        print()
    
    # Find clusters (groups with high similarity)
    threshold = 0.8
    clusters = []
    used_embeddings = set()
    
    for pair in similarities:
        if pair['similarity'] >= threshold:
            emb1_id = pair['emb1_id']
            emb2_id = pair['emb2_id']
            
            # Check if either embedding is already in a cluster
            found_cluster = False
            for cluster in clusters:
                if emb1_id in cluster or emb2_id in cluster:
                    cluster.add(emb1_id)
                    cluster.add(emb2_id)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters.append({emb1_id, emb2_id})
    
    if clusters:
        print(f"   Found {len(clusters)} similarity clusters (threshold: {threshold}):")
        for i, cluster in enumerate(clusters, 1):
            print(f"   Cluster {i} ({len(cluster)} items):")
            for emb_id in cluster:
                emb_data = next(e for e in all_embeddings if e['id'] == emb_id)
                print(f"      - '{emb_data['transcription']}'")
            print()
    else:
        print(f"   No clusters found with threshold {threshold}")

def export_similarity_results():
    """Export similarity search results to various formats."""
    
    print("\n📤 Exporting Similarity Results")
    print("=" * 50)
    
    query_engine = EmbeddingQueryEngine()
    
    # Export embeddings to numpy format
    print("📊 Exporting embeddings to numpy format...")
    try:
        export_file = query_engine.export_embeddings_to_numpy("similarity_analysis_embeddings.npz")
        print(f"   ✅ Exported to: {export_file}")
        
        # Load and verify export
        embeddings, metadata = query_engine.load_embeddings_from_numpy(export_file)
        print(f"   📈 Loaded {len(embeddings)} embeddings with metadata")
        
    except Exception as e:
        print(f"   ❌ Export failed: {e}")
    
    # Export similarity matrix
    print("\n📊 Exporting similarity matrix...")
    try:
        all_embeddings = query_engine.get_all_embeddings('transcript')
        
        if len(all_embeddings) > 1:
            # Create similarity matrix
            n = len(all_embeddings)
            similarity_matrix = np.zeros((n, n))
            
            for i, emb1 in enumerate(all_embeddings):
                for j, emb2 in enumerate(all_embeddings):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        similarity_matrix[i][j] = query_engine.calculate_cosine_similarity(
                            query_engine.get_embedding_array(emb1['id']),
                            query_engine.get_embedding_array(emb2['id'])
                        )
            
            # Save matrix
            np.savez("similarity_matrix.npz", 
                     matrix=similarity_matrix,
                     embeddings=[e['id'] for e in all_embeddings],
                     transcripts=[e['transcription'] for e in all_embeddings])
            
            print(f"   ✅ Similarity matrix exported to: similarity_matrix.npz")
            print(f"   📈 Matrix shape: {similarity_matrix.shape}")
            print(f"   📊 Average similarity: {np.mean(similarity_matrix):.3f}")
            
    except Exception as e:
        print(f"   ❌ Matrix export failed: {e}")

def main():
    """Main function to run similarity search examples."""
    
    print("🚀 WhisperPOC Similarity Search Examples")
    print("=" * 60)
    
    # Check if database is available
    try:
        storage = AudioStorageSystem(storage_backend="postgres")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please ensure PostgreSQL is running and database is created.")
        return
    
    # Check if embeddings exist
    try:
        query_engine = EmbeddingQueryEngine()
        stats = query_engine.get_embedding_statistics()
        
        if stats['total_embeddings'] == 0:
            print("❌ No embeddings found in database")
            print("Please record some audio files first to generate embeddings.")
            return
        
        print(f"✅ Found {stats['total_embeddings']} embeddings")
        
    except Exception as e:
        print(f"❌ Failed to initialize query engine: {e}")
        return
    
    # Run examples
    try:
        # Basic similarity search
        basic_similarity_search()
        
        # Metadata-filtered search
        metadata_filtered_search()
        
        # Advanced analysis
        advanced_similarity_analysis()
        
        # Similarity clustering
        similarity_clustering()
        
        # Export results
        export_similarity_results()
        
        print(f"\n🎉 Similarity search examples completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Similarity search interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during similarity search: {e}")

if __name__ == "__main__":
    main() 