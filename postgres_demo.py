#!/usr/bin/env python3
"""
PostgreSQL Audio Storage System Demo
Demonstrates advanced features and capabilities
"""

import time
import numpy as np
from audio_storage_system import AudioStorageSystem

def demo_postgres_features():
    """Demonstrate PostgreSQL features"""
    print("🗄️  PostgreSQL Audio Storage System Demo")
    print("=" * 60)
    
    # Initialize PostgreSQL storage
    storage = AudioStorageSystem(storage_backend="postgres")
    
    try:
        # 1. Batch recording and storage
        print("\n1. 📝 Batch Recording and Storage")
        print("-" * 40)
        
        audio_ids = []
        for i in range(3):
            print(f"\n   Recording audio {i+1}/3...")
            audio_id = storage.record_and_store_audio(
                duration=3,
                whisper_model="base",
                generate_embeddings=True,
                metadata={
                    "user": "demo_user",
                    "session": f"demo_session_{i+1}",
                    "batch_id": "batch_001",
                    "priority": "high" if i == 0 else "normal"
                }
            )
            audio_ids.append(audio_id)
            print(f"   ✅ Audio {i+1} stored with ID: {audio_id}")
        
        # 2. Advanced queries
        print("\n2. 🔍 Advanced PostgreSQL Queries")
        print("-" * 40)
        
        # Query by metadata
        print("\n   📊 Audio files by user:")
        storage.cursor.execute("""
            SELECT af.id, af.file_path, af.duration, m.value as user
            FROM audio_files af
            JOIN metadata m ON af.id = m.audio_file_id
            WHERE m.key = 'user' AND m.value = 'demo_user'
            ORDER BY af.created_at DESC
        """)
        user_audio = storage.cursor.fetchall()
        for row in user_audio:
            print(f"      ID: {row['id']}, Duration: {row['duration']:.1f}s, User: {row['user']}")
        
        # Query by session
        print("\n   📊 Audio files by session:")
        storage.cursor.execute("""
            SELECT af.id, af.file_path, af.duration, m.value as session
            FROM audio_files af
            JOIN metadata m ON af.id = m.audio_file_id
            WHERE m.key = 'session'
            ORDER BY af.created_at DESC
        """)
        session_audio = storage.cursor.fetchall()
        for row in session_audio:
            print(f"      ID: {row['id']}, Session: {row['session']}, Duration: {row['duration']:.1f}s")
        
        # 3. Complex search queries
        print("\n3. 🔍 Complex Search Queries")
        print("-" * 40)
        
        # Search with multiple criteria
        print("\n   🔍 Search for high priority audio:")
        storage.cursor.execute("""
            SELECT 
                af.id,
                af.file_path,
                af.duration,
                t.transcription,
                m1.value as user,
                m2.value as priority
            FROM audio_files af
            LEFT JOIN transcripts t ON af.id = t.audio_file_id
            LEFT JOIN metadata m1 ON af.id = m1.audio_file_id AND m1.key = 'user'
            LEFT JOIN metadata m2 ON af.id = m2.audio_file_id AND m2.key = 'priority'
            WHERE m2.value = 'high'
            ORDER BY af.created_at DESC
        """)
        high_priority = storage.cursor.fetchall()
        for row in high_priority:
            print(f"      ID: {row['id']}, User: {row['user']}, Priority: {row['priority']}")
            print(f"      Transcript: {row['transcription'][:50]}...")
        
        # 4. Data aggregation
        print("\n4. 📊 Data Aggregation")
        print("-" * 40)
        
        # Total storage usage
        storage.cursor.execute("""
            SELECT 
                COUNT(*) as total_files,
                SUM(file_size) as total_size_bytes,
                AVG(duration) as avg_duration,
                MIN(created_at) as earliest_recording,
                MAX(created_at) as latest_recording
            FROM audio_files
        """)
        summary = storage.cursor.fetchone()
        
        print(f"\n   📊 Storage Summary:")
        print(f"      Total files: {summary['total_files']}")
        print(f"      Total size: {summary['total_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"      Average duration: {summary['avg_duration']:.1f} seconds")
        print(f"      Date range: {summary['earliest_recording']} to {summary['latest_recording']}")
        
        # 5. Similarity search with PostgreSQL
        print("\n5. 🧠 Similarity Search")
        print("-" * 40)
        
        # Search for similar audio
        results = storage.search_similar_audio("demo recording", limit=3)
        print(f"\n   🔍 Similar audio results:")
        for i, result in enumerate(results, 1):
            print(f"      {i}. Similarity: {result['similarity']:.3f}")
            print(f"         Transcript: {result['transcription'][:50]}...")
        
        # 6. Database maintenance
        print("\n6. 🛠️  Database Maintenance")
        print("-" * 40)
        
        # Analyze tables for better query planning
        storage.cursor.execute("ANALYZE audio_files")
        storage.cursor.execute("ANALYZE transcripts")
        storage.cursor.execute("ANALYZE embeddings")
        storage.cursor.execute("ANALYZE metadata")
        storage.conn.commit()
        print("   ✅ Database tables analyzed for optimal performance")
        
        # Check table sizes
        storage.cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
        tables = storage.cursor.fetchall()
        
        print("\n   📊 Table Sizes:")
        for table in tables:
            print(f"      {table['tablename']}: {table['size']}")
        
        # 7. Export capabilities
        print("\n7. 📤 Export Capabilities")
        print("-" * 40)
        
        # Export to CSV
        import csv
        csv_file = "audio_export.csv"
        
        storage.cursor.execute("""
            SELECT 
                af.id,
                af.file_path,
                af.duration,
                af.file_size,
                t.transcription,
                t.confidence,
                af.created_at
            FROM audio_files af
            LEFT JOIN transcripts t ON af.id = t.audio_file_id
            ORDER BY af.created_at DESC
        """)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['ID', 'File Path', 'Duration', 'File Size', 'Transcription', 'Confidence', 'Created At'])
            # Write data
            for row in storage.cursor.fetchall():
                writer.writerow([
                    row['id'],
                    row['file_path'],
                    row['duration'],
                    row['file_size'],
                    row['transcription'],
                    row['confidence'],
                    row['created_at']
                ])
        
        print(f"   ✅ Data exported to {csv_file}")
        
        # 8. Cleanup demonstration
        print("\n8. 🧹 Cleanup Demonstration")
        print("-" * 40)
        
        # Show cleanup without actually deleting (dry run)
        storage.cursor.execute("""
            SELECT COUNT(*) as old_files
            FROM audio_files 
            WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day'
        """)
        old_count = storage.cursor.fetchone()['old_files']
        print(f"   📊 Files older than 1 day: {old_count}")
        print("   💡 Use storage.cleanup_old_files(days_old=1) to clean up old files")
        
        # 9. Show all data in database
        print("\n9. 📋 Complete Database Overview")
        print("-" * 40)
        
        # Show all audio files with their metadata
        storage.cursor.execute("""
            SELECT 
                af.id,
                af.file_path,
                af.duration,
                af.file_size,
                t.transcription,
                t.confidence,
                af.created_at,
                string_agg(m.key || ':' || m.value, ', ') as metadata
            FROM audio_files af
            LEFT JOIN transcripts t ON af.id = t.audio_file_id
            LEFT JOIN metadata m ON af.id = m.audio_file_id
            GROUP BY af.id, af.file_path, af.duration, af.file_size, 
                     t.transcription, t.confidence, af.created_at
            ORDER BY af.created_at DESC
        """)
        
        all_files = storage.cursor.fetchall()
        print(f"\n   📊 All Audio Files ({len(all_files)} total):")
        for file in all_files:
            print(f"      ID: {file['id']}")
            print(f"         Path: {file['file_path']}")
            print(f"         Duration: {file['duration']:.1f}s, Size: {file['file_size']} bytes")
            print(f"         Transcript: {file['transcription'][:50]}...")
            print(f"         Confidence: {file['confidence']:.3f}")
            print(f"         Metadata: {file['metadata']}")
            print()
        
    finally:
        # Close database connection
        storage.conn.close()
        print("\n🔒 Database connection closed")

if __name__ == "__main__":
    demo_postgres_features() 