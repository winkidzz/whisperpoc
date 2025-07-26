#!/usr/bin/env python3
"""
Interactive Audio Recorder with PostgreSQL Storage
Record multiple audio files with custom metadata
"""

import time
import sys
from audio_storage_system import AudioStorageSystem

def get_user_input(prompt, default=""):
    """Get user input with optional default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def interactive_recorder():
    """Interactive audio recording session"""
    print("🎤 Interactive Audio Recorder with PostgreSQL Storage")
    print("=" * 60)
    
    # Initialize PostgreSQL storage
    storage = AudioStorageSystem(storage_backend="postgres")
    
    try:
        while True:
            print("\n" + "="*50)
            print("🎵 NEW RECORDING SESSION")
            print("="*50)
            
            # Get recording parameters
            duration = int(get_user_input("Recording duration (seconds)", "10"))
            whisper_model = get_user_input("Whisper model", "base")
            
            # Get metadata
            print("\n📝 Metadata (press Enter to skip):")
            user = get_user_input("User name", "")
            session = get_user_input("Session name", "")
            priority = get_user_input("Priority (high/normal/low)", "normal")
            tags = get_user_input("Tags (comma-separated)", "")
            notes = get_user_input("Notes", "")
            
            # Build metadata dictionary
            metadata = {}
            if user:
                metadata["user"] = user
            if session:
                metadata["session"] = session
            if priority:
                metadata["priority"] = priority
            if tags:
                metadata["tags"] = tags
            if notes:
                metadata["notes"] = notes
            
            # Add timestamp
            metadata["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n🎤 Ready to record {duration} seconds...")
            print("📝 Metadata:", metadata)
            
            # Confirm recording
            confirm = get_user_input("Start recording? (y/n)", "y").lower()
            if confirm not in ['y', 'yes']:
                print("❌ Recording cancelled")
                continue
            
            # Record audio
            print(f"\n🎤 Recording {duration} seconds...")
            print("   Speak now!")
            
            audio_id = storage.record_and_store_audio(
                duration=duration,
                whisper_model=whisper_model,
                generate_embeddings=True,
                metadata=metadata
            )
            
            print(f"✅ Audio recorded and stored with ID: {audio_id}")
            
            # Show recording details
            audio_info = storage.get_audio_info(audio_id)
            if audio_info:
                print(f"\n📊 Recording Details:")
                print(f"   📁 File: {audio_info['audio_file']['file_path']}")
                print(f"   ⏱️  Duration: {audio_info['audio_file']['duration']:.1f} seconds")
                print(f"   📏 Size: {audio_info['audio_file']['file_size']} bytes")
                print(f"   🎯 Transcript: '{audio_info['transcript']['transcription']}'")
                print(f"   🎯 Confidence: {audio_info['transcript']['confidence']:.3f}")
                print(f"   🏷️  Metadata: {audio_info['metadata']}")
            
            # Ask if user wants to continue
            continue_recording = get_user_input("\nRecord another audio? (y/n)", "y").lower()
            if continue_recording not in ['y', 'yes']:
                break
        
        # Show summary
        print("\n" + "="*50)
        print("📊 RECORDING SESSION SUMMARY")
        print("="*50)
        
        all_audio = storage.list_all_audio(limit=100)
        print(f"📁 Total audio files in database: {len(all_audio)}")
        
        # Show recent recordings
        print(f"\n🎵 Recent Recordings:")
        for i, audio in enumerate(all_audio[:5], 1):
            print(f"   {i}. ID: {audio['id']}, Duration: {audio['duration']:.1f}s")
            if 'transcription' in audio and audio['transcription']:
                print(f"      Transcript: {audio['transcription'][:50]}...")
        
        # Show storage statistics
        storage.cursor.execute("""
            SELECT 
                COUNT(*) as total_files,
                SUM(file_size) as total_size_bytes,
                AVG(duration) as avg_duration
            FROM audio_files
        """)
        stats = storage.cursor.fetchone()
        
        print(f"\n📈 Storage Statistics:")
        print(f"   📁 Total files: {stats['total_files']}")
        print(f"   💾 Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"   ⏱️  Average duration: {stats['avg_duration']:.1f} seconds")
        
        # Export option
        export = get_user_input("\nExport data to CSV? (y/n)", "n").lower()
        if export in ['y', 'yes']:
            import csv
            csv_file = f"audio_export_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            
            storage.cursor.execute("""
                SELECT 
                    af.id,
                    af.file_path,
                    af.duration,
                    af.file_size,
                    t.transcription,
                    t.confidence,
                    af.created_at,
                    string_agg(m.key || ':' || m.value, '; ') as metadata
                FROM audio_files af
                LEFT JOIN transcripts t ON af.id = t.audio_file_id
                LEFT JOIN metadata m ON af.id = m.audio_file_id
                GROUP BY af.id, af.file_path, af.duration, af.file_size, 
                         t.transcription, t.confidence, af.created_at
                ORDER BY af.created_at DESC
            """)
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'File Path', 'Duration', 'File Size', 'Transcription', 'Confidence', 'Created At', 'Metadata'])
                for row in storage.cursor.fetchall():
                    writer.writerow([
                        row['id'],
                        row['file_path'],
                        row['duration'],
                        row['file_size'],
                        row['transcription'],
                        row['confidence'],
                        row['created_at'],
                        row['metadata']
                    ])
            
            print(f"✅ Data exported to {csv_file}")
        
        print("\n🎉 Recording session completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Recording session interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Close database connection
        storage.conn.close()
        print("🔒 Database connection closed")

if __name__ == "__main__":
    interactive_recorder() 