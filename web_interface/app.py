#!/usr/bin/env python3
"""
WhisperPOC Web Interface

A Flask-based web application providing a browser interface for:
- Audio recording and transcription
- Viewing and searching audio files
- Managing metadata and embeddings
- Real-time transcription display
"""

import os
import sys
import json
import time
import base64
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import queue

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_storage_system import AudioStorageSystem
from embedding_queries import EmbeddingQueryEngine
from config import get_postgres_config, get_whisper_config, print_config_summary

app = Flask(__name__)
app.config['SECRET_KEY'] = 'whisperpoc-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Helper function for formatting file sizes
def format_file_size(bytes):
    if bytes == 0:
        return '0 Bytes'
    k = 1024
    sizes = ['Bytes', 'KB', 'MB', 'GB']
    i = int(math.floor(math.log(bytes) / math.log(k)))
    return f"{bytes / math.pow(k, i):.1f} {sizes[i]}"

# Add template filter
app.jinja_env.filters['formatFileSize'] = format_file_size

# Initialize storage system
try:
    storage = AudioStorageSystem(storage_backend="postgres")
    query_engine = EmbeddingQueryEngine()
    print("✅ Web interface initialized with database connection")
except Exception as e:
    print(f"❌ Failed to initialize storage: {e}")
    storage = None
    query_engine = None

# Global variables for real-time recording
recording_queue = queue.Queue()
is_recording = False
recording_thread = None

@app.route('/')
def index():
    """Main dashboard page."""
    if not storage:
        return render_template('error.html', error="Database connection failed")
    
    try:
        # Get basic statistics
        audio_count = storage.get_audio_count()
        transcript_count = storage.get_transcript_count()
        embedding_count = storage.get_embedding_count()
        
        # Get recent recordings
        recent_audio = storage.list_all_audio(limit=10)
        
        return render_template('dashboard.html',
                             audio_count=audio_count,
                             transcript_count=transcript_count,
                             embedding_count=embedding_count,
                             recent_audio=recent_audio,
                             formatFileSize=format_file_size)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/record')
def record_page():
    """Audio recording page."""
    return render_template('record.html')

@app.route('/browse')
def browse_page():
    """Browse and search audio files."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = 20
        offset = (page - 1) * limit
        
        audio_files = storage.list_all_audio(limit=limit, offset=offset)
        total_count = storage.get_audio_count()
        total_pages = (total_count + limit - 1) // limit
        
        return render_template('browse.html',
                             audio_files=audio_files,
                             current_page=page,
                             total_pages=total_pages,
                             total_count=total_count,
                             formatFileSize=format_file_size)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/search')
def search_page():
    """Search page."""
    return render_template('search.html')

@app.route('/api/record', methods=['POST'])
def api_record_audio():
    """API endpoint for recording audio."""
    try:
        data = request.get_json()
        duration = data.get('duration', 10)
        whisper_model = data.get('whisper_model', 'base')
        metadata = data.get('metadata', {})
        
        # Add timestamp to metadata
        metadata['recorded_at'] = datetime.now().isoformat()
        metadata['source'] = 'web_interface'
        
        # Record audio
        audio_id = storage.record_and_store_audio(
            duration=duration,
            whisper_model=whisper_model,
            generate_embeddings=True,
            metadata=metadata
        )
        
        # Get audio info
        audio_info = storage.get_audio_info(audio_id)
        
        return jsonify({
            'success': True,
            'audio_id': audio_id,
            'audio_info': audio_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/audio/<int:audio_id>')
def api_get_audio_info(audio_id):
    """Get audio file information."""
    try:
        audio_info = storage.get_audio_info(audio_id)
        return jsonify({
            'success': True,
            'audio_info': audio_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for similar audio content."""
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        limit = data.get('limit', 10)
        
        if not query_text:
            return jsonify({
                'success': False,
                'error': 'Query text is required'
            }), 400
        
        results = query_engine.search_by_text_similarity(
            query_text=query_text,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/audio/<int:audio_id>/download')
def api_download_audio(audio_id):
    """Download audio file."""
    try:
        audio_info = storage.get_audio_info(audio_id)
        file_path = audio_info['audio_file']['file_path']
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': 'Audio file not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/audio/<int:audio_id>/delete', methods=['DELETE'])
def api_delete_audio(audio_id):
    """Delete audio file."""
    try:
        # Get audio info first
        audio_info = storage.get_audio_info(audio_id)
        file_path = audio_info['audio_file']['file_path']
        
        # Delete from database (cascade will handle related records)
        storage.cursor.execute("DELETE FROM audio_files WHERE id = %s", (audio_id,))
        storage.conn.commit()
        
        # Delete file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'success': True,
            'message': f'Audio file {audio_id} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics')
def api_get_statistics():
    """Get system statistics."""
    try:
        # Basic statistics
        audio_count = storage.get_audio_count()
        transcript_count = storage.get_transcript_count()
        embedding_count = storage.get_embedding_count()
        
        # Embedding statistics
        embedding_stats = query_engine.get_embedding_statistics()
        
        # Recent activity
        storage.cursor.execute("""
            SELECT COUNT(*) as today_count
            FROM audio_files
            WHERE DATE(created_at) = CURRENT_DATE
        """)
        today_count = storage.cursor.fetchone()['today_count']
        
        # Average confidence
        storage.cursor.execute("""
            SELECT AVG(confidence) as avg_confidence
            FROM transcripts
            WHERE confidence IS NOT NULL
        """)
        avg_confidence = storage.cursor.fetchone()['avg_confidence'] or 0
        
        return jsonify({
            'success': True,
            'statistics': {
                'audio_count': audio_count,
                'transcript_count': transcript_count,
                'embedding_count': embedding_count,
                'today_count': today_count,
                'avg_confidence': round(avg_confidence, 3),
                'embedding_stats': embedding_stats
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export')
def api_export_data():
    """Export data in various formats."""
    try:
        format_type = request.args.get('format', 'json')
        
        if format_type == 'json':
            # Export as JSON
            audio_files = storage.list_all_audio(limit=1000)
            export_data = []
            
            for audio in audio_files:
                try:
                    audio_info = storage.get_audio_info(audio['id'])
                    export_data.append(audio_info)
                except:
                    continue
            
            return jsonify({
                'success': True,
                'data': export_data,
                'format': 'json'
            })
            
        elif format_type == 'csv':
            # Export as CSV
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
            
            # Convert to CSV format
            csv_data = "id,file_path,duration,file_size,transcription,confidence,language,created_at\n"
            for row in data:
                csv_data += f"{row['id']},{row['file_path']},{row['duration']},{row['file_size']},\"{row['transcription'] or ''}\",{row['confidence'] or 0},{row['language'] or ''},{row['created_at']}\n"
            
            return jsonify({
                'success': True,
                'data': csv_data,
                'format': 'csv'
            })
            
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported format'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WebSocket events for real-time features
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'message': 'Connected to WhisperPOC Web Interface'})

@socketio.on('start_recording')
def handle_start_recording(data):
    """Start real-time recording."""
    global is_recording, recording_thread
    
    if is_recording:
        emit('error', {'message': 'Recording already in progress'})
        return
    
    duration = data.get('duration', 30)
    whisper_model = data.get('whisper_model', 'base')
    
    is_recording = True
    recording_thread = threading.Thread(
        target=real_time_recording,
        args=(duration, whisper_model)
    )
    recording_thread.start()
    
    emit('recording_started', {'duration': duration})

@socketio.on('stop_recording')
def handle_stop_recording():
    """Stop real-time recording."""
    global is_recording
    is_recording = False
    emit('recording_stopped', {'message': 'Recording stopped'})

def real_time_recording(duration, whisper_model):
    """Real-time recording function."""
    global is_recording
    
    try:
        # Record audio in chunks
        chunk_duration = 5  # 5-second chunks
        chunks = []
        
        for i in range(0, duration, chunk_duration):
            if not is_recording:
                break
                
            # Record chunk
            chunk_duration_actual = min(chunk_duration, duration - i)
            
            # Simulate recording (in real implementation, this would record audio)
            time.sleep(chunk_duration_actual)
            
            # Emit progress
            progress = min(100, (i + chunk_duration_actual) / duration * 100)
            socketio.emit('recording_progress', {
                'progress': progress,
                'time_elapsed': i + chunk_duration_actual
            })
        
        if is_recording:
            # Process complete recording
            socketio.emit('recording_complete', {
                'message': 'Recording completed, processing...'
            })
            
            # Here you would process the recorded audio
            # For now, we'll simulate it
            time.sleep(2)
            
            socketio.emit('processing_complete', {
                'message': 'Audio processed successfully'
            })
    
    except Exception as e:
        socketio.emit('error', {'message': f'Recording error: {str(e)}'})
    finally:
        is_recording = False

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        if storage:
            # Test database connection
            storage.cursor.execute("SELECT 1")
            db_status = "healthy"
        else:
            db_status = "unhealthy"
        
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Print configuration summary
    print("=" * 60)
    print("WhisperPOC Web Interface")
    print("=" * 60)
    print_config_summary()
    
    # Start the web server
    print("\n🚀 Starting web interface...")
    print("📱 Open your browser to: http://localhost:5002")
    print("🔧 API documentation available at: http://localhost:5002/api")
    print("💚 Health check: http://localhost:5002/health")
    
    socketio.run(app, 
                host='0.0.0.0', 
                port=5002, 
                debug=True,
                allow_unsafe_werkzeug=True) 