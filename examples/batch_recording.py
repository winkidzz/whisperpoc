#!/usr/bin/env python3
"""
Batch Recording Example

This script demonstrates how to record multiple audio files with different
settings and metadata in a batch operation.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_storage_system import AudioStorageSystem
from config import get_audio_config, get_whisper_config

def batch_recording_example():
    """Example of batch recording with different settings."""
    
    print("🎤 Batch Recording Example")
    print("=" * 50)
    
    # Initialize storage system
    storage = AudioStorageSystem(storage_backend="postgres")
    
    # Define batch recording scenarios
    recording_scenarios = [
        {
            'name': 'Quick Note',
            'duration': 5,
            'whisper_model': 'tiny',
            'metadata': {
                'user': 'john',
                'session': 'quick_notes',
                'priority': 'low',
                'tags': 'note,quick',
                'notes': 'Quick voice note for later reference'
            }
        },
        {
            'name': 'Meeting Recording',
            'duration': 30,
            'whisper_model': 'base',
            'metadata': {
                'user': 'john',
                'session': 'team_meeting',
                'priority': 'high',
                'tags': 'meeting,team,important',
                'notes': 'Team standup meeting recording'
            }
        },
        {
            'name': 'Interview Session',
            'duration': 60,
            'whisper_model': 'small',
            'metadata': {
                'user': 'john',
                'session': 'interview',
                'priority': 'high',
                'tags': 'interview,candidate,important',
                'notes': 'Candidate interview recording'
            }
        },
        {
            'name': 'Presentation Practice',
            'duration': 120,
            'whisper_model': 'medium',
            'metadata': {
                'user': 'john',
                'session': 'presentation',
                'priority': 'normal',
                'tags': 'presentation,practice',
                'notes': 'Practice run for upcoming presentation'
            }
        }
    ]
    
    recorded_files = []
    
    for i, scenario in enumerate(recording_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['name']}")
        print(f"   Duration: {scenario['duration']} seconds")
        print(f"   Whisper Model: {scenario['whisper_model']}")
        print(f"   Priority: {scenario['metadata']['priority']}")
        
        # Ask for confirmation
        confirm = input(f"\nStart recording '{scenario['name']}'? (y/n): ").lower()
        if confirm not in ['y', 'yes']:
            print("   ⏭️  Skipped")
            continue
        
        print(f"\n🎤 Recording {scenario['duration']} seconds...")
        print("   Speak now!")
        
        try:
            # Record and store audio
            audio_id = storage.record_and_store_audio(
                duration=scenario['duration'],
                whisper_model=scenario['whisper_model'],
                generate_embeddings=True,
                metadata=scenario['metadata']
            )
            
            # Get recording info
            audio_info = storage.get_audio_info(audio_id)
            
            recorded_files.append({
                'id': audio_id,
                'name': scenario['name'],
                'transcription': audio_info['transcript']['transcription'],
                'confidence': audio_info['transcript']['confidence'],
                'duration': audio_info['audio_file']['duration']
            })
            
            print(f"   ✅ Recorded successfully!")
            print(f"   📝 Transcript: '{audio_info['transcript']['transcription']}'")
            print(f"   🎯 Confidence: {audio_info['transcript']['confidence']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Recording failed: {e}")
            continue
        
        # Brief pause between recordings
        if i < len(recording_scenarios):
            print("\n⏳ Pausing 3 seconds before next recording...")
            time.sleep(3)
    
    # Summary
    print(f"\n📊 Batch Recording Summary")
    print("=" * 50)
    print(f"Total scenarios: {len(recording_scenarios)}")
    print(f"Successfully recorded: {len(recorded_files)}")
    
    if recorded_files:
        print(f"\n📋 Recorded Files:")
        for file_info in recorded_files:
            print(f"   🎵 {file_info['name']}")
            print(f"      ID: {file_info['id']}")
            print(f"      Duration: {file_info['duration']:.1f}s")
            print(f"      Confidence: {file_info['confidence']:.2f}")
            print(f"      Transcript: '{file_info['transcription']}'")
            print()
    
    # Export summary
    export_choice = input("Export recording summary to CSV? (y/n): ").lower()
    if export_choice in ['y', 'yes']:
        export_batch_summary(storage, recorded_files)
    
    return recorded_files

def export_batch_summary(storage, recorded_files):
    """Export batch recording summary to CSV."""
    
    import csv
    from datetime import datetime
    
    filename = f"batch_recording_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'name', 'transcription', 'confidence', 'duration', 'created_at']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file_info in recorded_files:
            # Get additional info from database
            audio_info = storage.get_audio_info(file_info['id'])
            writer.writerow({
                'id': file_info['id'],
                'name': file_info['name'],
                'transcription': file_info['transcription'],
                'confidence': file_info['confidence'],
                'duration': file_info['duration'],
                'created_at': audio_info['audio_file']['created_at']
            })
    
    print(f"📄 Summary exported to: {filename}")

def batch_recording_with_metadata():
    """Example of recording with dynamic metadata generation."""
    
    print("\n🎯 Advanced Batch Recording with Dynamic Metadata")
    print("=" * 60)
    
    storage = AudioStorageSystem(storage_backend="postgres")
    
    # Generate metadata based on time and context
    current_time = time.strftime("%H:%M")
    current_date = time.strftime("%Y-%m-%d")
    
    # Morning recordings
    if current_time < "12:00":
        time_context = "morning"
        session_type = "morning_notes"
    elif current_time < "17:00":
        time_context = "afternoon"
        session_type = "afternoon_work"
    else:
        time_context = "evening"
        session_type = "evening_reflection"
    
    # Record multiple quick notes
    for i in range(3):
        metadata = {
            'user': 'john',
            'session': session_type,
            'time_context': time_context,
            'date': current_date,
            'note_number': i + 1,
            'priority': 'normal',
            'tags': f'note,{time_context},quick',
            'notes': f'Quick {time_context} note #{i+1}'
        }
        
        print(f"\n📝 Recording {time_context} note #{i+1}")
        print(f"   Session: {session_type}")
        print(f"   Date: {current_date}")
        
        try:
            audio_id = storage.record_and_store_audio(
                duration=10,
                whisper_model='tiny',
                generate_embeddings=True,
                metadata=metadata
            )
            
            audio_info = storage.get_audio_info(audio_id)
            print(f"   ✅ Recorded: '{audio_info['transcript']['transcription']}'")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        if i < 2:  # Don't pause after the last recording
            time.sleep(2)

def main():
    """Main function to run batch recording examples."""
    
    print("🚀 WhisperPOC Batch Recording Examples")
    print("=" * 60)
    
    # Check if database is available
    try:
        storage = AudioStorageSystem(storage_backend="postgres")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please ensure PostgreSQL is running and database is created.")
        return
    
    # Run examples
    try:
        # Basic batch recording
        recorded_files = batch_recording_example()
        
        # Advanced batch recording with dynamic metadata
        batch_recording_with_metadata()
        
        print(f"\n🎉 Batch recording examples completed!")
        print(f"Check your PostgreSQL database for the recorded files.")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Batch recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during batch recording: {e}")

if __name__ == "__main__":
    main() 