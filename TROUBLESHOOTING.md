# WhisperPOC Troubleshooting Guide

## 🔧 **Quick Diagnosis**

### **System Health Check**
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check PostgreSQL
psql --version

# Check audio devices
python test_audio.py

# Check database connection
python -c "from audio_storage_system import AudioStorageSystem; AudioStorageSystem(storage_backend='postgres')"
```

## 🚨 **Common Issues & Solutions**

### **1. Audio Recording Issues**

#### **Problem: "No audio input detected"**
**Symptoms:**
- No audio is recorded
- Silent recordings
- Error messages about no audio input

**Solutions:**
```bash
# Check microphone permissions (macOS)
System Preferences > Security & Privacy > Microphone > Enable for Terminal/Python

# Check audio devices
python test_audio.py --list-devices

# Test microphone manually
python -c "
import sounddevice as sd
import numpy as np
print('Recording 3 seconds...')
audio = sd.rec(3 * 16000, samplerate=16000, channels=1)
sd.wait()
print(f'Audio recorded: {len(audio)} samples')
print(f'Max amplitude: {np.max(np.abs(audio))}')
"
```

**Additional Checks:**
- Ensure microphone is not muted
- Check system volume settings
- Try different audio input devices
- Restart audio services

#### **Problem: "PortAudio not found"**
**Symptoms:**
- `fatal error: 'portaudio.h' file not found`
- PyAudio installation fails

**Solutions:**
```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

#### **Problem: Audio quality issues**
**Symptoms:**
- Poor transcription accuracy
- Audio distortion
- Low volume recordings

**Solutions:**
```python
# Adjust audio settings
from audio_storage_system import AudioStorageSystem

storage = AudioStorageSystem()
# Use higher sample rate for better quality
audio_id = storage.record_and_store_audio(
    duration=10,
    sample_rate=44100,  # Higher quality
    channels=1
)
```

### **2. Database Issues**

#### **Problem: "Connection refused"**
**Symptoms:**
- `psycopg2.OperationalError: connection to server failed`
- Database connection errors

**Solutions:**
```bash
# Check PostgreSQL status
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql      # Ubuntu

# Start PostgreSQL if stopped
brew services start postgresql        # macOS
sudo systemctl start postgresql       # Ubuntu

# Test connection
psql -d audio_storage_db -c "SELECT version();"

# Create database if it doesn't exist
createdb audio_storage_db
```

#### **Problem: "Database does not exist"**
**Symptoms:**
- `psycopg2.OperationalError: database "audio_storage_db" does not exist`

**Solutions:**
```bash
# Create database
createdb audio_storage_db

# Or using psql
psql -U postgres
CREATE DATABASE audio_storage_db;
\q
```

#### **Problem: Permission denied**
**Symptoms:**
- `psycopg2.OperationalError: permission denied for database`

**Solutions:**
```bash
# Grant permissions
psql -U postgres -d audio_storage_db
GRANT ALL PRIVILEGES ON DATABASE audio_storage_db TO your_username;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_username;
\q
```

### **3. Whisper Model Issues**

#### **Problem: "Model download failed"**
**Symptoms:**
- `whisper.load_model()` fails
- Network timeout errors

**Solutions:**
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper
python -c "import whisper; whisper.load_model('base')"

# Check internet connection
curl -I https://openaipublic.azureedge.net

# Use smaller model for testing
python -c "import whisper; whisper.load_model('tiny')"
```

#### **Problem: "Out of memory"**
**Symptoms:**
- Memory errors when loading large models
- System becomes unresponsive

**Solutions:**
```python
# Use smaller models
storage.record_and_store_audio(
    duration=10,
    whisper_model="tiny",  # Smallest model
    generate_embeddings=False  # Disable embeddings temporarily
)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### **Problem: Slow transcription**
**Symptoms:**
- Long processing times
- High CPU usage

**Solutions:**
```python
# Use GPU if available
import torch
if torch.cuda.is_available():
    print("GPU available for Whisper")
    # Whisper will automatically use GPU

# Use smaller models for speed
whisper_models = ["tiny", "base", "small", "medium", "large"]
# tiny = fastest, large = slowest
```

### **4. Embedding Issues**

#### **Problem: "No embeddings found"**
**Symptoms:**
- Empty embedding queries
- No similarity search results

**Solutions:**
```python
# Check if embeddings are being generated
from embedding_queries import EmbeddingQueryEngine

query_engine = EmbeddingQueryEngine()
stats = query_engine.get_embedding_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")

# Generate embeddings for existing audio
from audio_storage_system import AudioStorageSystem
storage = AudioStorageSystem()

# Re-generate embeddings for all audio files
audio_files = storage.list_all_audio()
for audio in audio_files:
    storage.generate_embeddings_for_audio(audio['id'])
```

#### **Problem: Embedding generation fails**
**Symptoms:**
- Errors during embedding generation
- Missing embedding data

**Solutions:**
```python
# Check embedding configuration
from config import get_embedding_config
config = get_embedding_config()
print(f"Embedding config: {config}")

# Test embedding generation manually
def test_embedding_generation():
    import numpy as np
    # Generate dummy embedding
    dummy_embedding = np.random.rand(1536)
    print(f"Generated embedding shape: {dummy_embedding.shape}")
    return dummy_embedding

# Override embedding function for testing
storage.generate_text_embedding = test_embedding_generation
```

### **5. Performance Issues**

#### **Problem: Slow database queries**
**Symptoms:**
- Long response times
- High database CPU usage

**Solutions:**
```sql
-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_audio_files_created_at ON audio_files(created_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_confidence ON transcripts(confidence);
CREATE INDEX IF NOT EXISTS idx_metadata_key_value ON metadata(key, value);

-- Analyze tables for query optimization
ANALYZE audio_files;
ANALYZE transcripts;
ANALYZE embeddings;
ANALYZE metadata;

-- Check query performance
EXPLAIN ANALYZE SELECT * FROM audio_files ORDER BY created_at DESC LIMIT 10;
```

#### **Problem: High memory usage**
**Symptoms:**
- System becomes slow
- Memory errors

**Solutions:**
```python
# Monitor memory usage
import psutil
import gc

def monitor_memory():
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent}%")
    print(f"Available: {memory.available / 1024**3:.1f} GB")
    
    if memory.percent > 80:
        print("High memory usage detected!")
        gc.collect()  # Force garbage collection

# Use generators for large datasets
def audio_file_generator():
    storage = AudioStorageSystem()
    for audio in storage.list_all_audio():
        yield audio
        monitor_memory()
```

### **6. Configuration Issues**

#### **Problem: Environment variables not loaded**
**Symptoms:**
- Default configuration used
- API keys not found

**Solutions:**
```bash
# Create .env file
cat > .env << EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=audio_storage_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

WHISPER_MODEL=base
DEBUG=true
EOF

# Load environment variables
export $(cat .env | xargs)
```

#### **Problem: Configuration validation fails**
**Symptoms:**
- Configuration errors on startup
- Invalid settings

**Solutions:**
```python
# Validate configuration
from config import validate_config, print_config_summary

if not validate_config():
    print("Configuration errors detected!")
    print_config_summary()
    
# Check specific settings
from config import get_postgres_config, get_whisper_config

postgres_config = get_postgres_config()
print(f"PostgreSQL config: {postgres_config}")

whisper_config = get_whisper_config()
print(f"Whisper config: {whisper_config}")
```

## 🔍 **Debugging Tools**

### **Debug Mode**
```bash
# Enable debug logging
export DEBUG=1
export LOG_LEVEL=DEBUG

# Run with debug output
python audio_storage_system.py --debug
```

### **Database Diagnostics**
```sql
-- Check database health
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check for orphaned records
SELECT 'orphaned_transcripts' as type, COUNT(*) as count
FROM transcripts t
LEFT JOIN audio_files af ON t.audio_file_id = af.id
WHERE af.id IS NULL
UNION ALL
SELECT 'orphaned_embeddings' as type, COUNT(*) as count
FROM embeddings e
LEFT JOIN audio_files af ON e.audio_file_id = af.id
WHERE af.id IS NULL;
```

### **Audio Diagnostics**
```python
# Test audio system
def test_audio_system():
    import sounddevice as sd
    import numpy as np
    
    print("Testing audio system...")
    
    # List devices
    devices = sd.query_devices()
    print(f"Found {len(devices)} audio devices")
    
    # Test recording
    print("Recording 3 seconds...")
    audio = sd.rec(3 * 16000, samplerate=16000, channels=1)
    sd.wait()
    
    print(f"Audio recorded: {len(audio)} samples")
    print(f"Max amplitude: {np.max(np.abs(audio))}")
    print(f"Mean amplitude: {np.mean(np.abs(audio))}")
    
    if np.max(np.abs(audio)) < 0.01:
        print("WARNING: Very low audio levels detected!")
    else:
        print("Audio recording successful!")

test_audio_system()
```

### **Performance Monitoring**
```python
# Monitor system performance
import time
import psutil
import threading

def performance_monitor():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        print(f"CPU: {cpu_percent}% | Memory: {memory_percent}% | Disk: {disk_usage}%")
        time.sleep(5)

# Start monitoring in background
monitor_thread = threading.Thread(target=performance_monitor, daemon=True)
monitor_thread.start()
```

## 🛠️ **Recovery Procedures**

### **Database Recovery**
```bash
# Backup database
pg_dump audio_storage_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
psql audio_storage_db < backup_20240101_120000.sql

# Reset database (WARNING: This will delete all data!)
dropdb audio_storage_db
createdb audio_storage_db
```

### **Audio File Recovery**
```python
# Re-scan audio files
def rescan_audio_files():
    import os
    from audio_storage_system import AudioStorageSystem
    
    storage = AudioStorageSystem()
    audio_dir = "audio_storage"
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(audio_dir, filename)
            try:
                storage.store_audio_file(file_path)
                print(f"Re-added: {filename}")
            except Exception as e:
                print(f"Failed to add {filename}: {e}")

rescan_audio_files()
```

### **Configuration Reset**
```python
# Reset to default configuration
import os

# Remove environment variables
env_vars = [
    'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
    'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY',
    'WHISPER_MODEL', 'DEBUG', 'LOG_LEVEL'
]

for var in env_vars:
    if var in os.environ:
        del os.environ[var]

# Re-import configuration
from config import print_config_summary
print_config_summary()
```

## 📞 **Getting Help**

### **Log Files**
```bash
# Check application logs
tail -f whisperpoc.log

# Check system logs
tail -f /var/log/syslog  # Ubuntu
log show --predicate 'process == "postgres"' --last 1h  # macOS
```

### **System Information**
```bash
# Collect system info for debugging
echo "=== System Information ===" > debug_info.txt
uname -a >> debug_info.txt
python3 --version >> debug_info.txt
psql --version >> debug_info.txt
brew list | grep -E "(postgresql|portaudio)" >> debug_info.txt
echo "=== Python Packages ===" >> debug_info.txt
pip list >> debug_info.txt
echo "=== Environment Variables ===" >> debug_info.txt
env | grep -E "(POSTGRES|OPENAI|ANTHROPIC|GOOGLE|WHISPER)" >> debug_info.txt
```

### **Common Error Codes**
- `EACCES`: Permission denied (check file/database permissions)
- `ECONNREFUSED`: Connection refused (check service status)
- `ENOMEM`: Out of memory (reduce model size or batch size)
- `ENOENT`: File not found (check file paths)
- `EINVAL`: Invalid argument (check configuration parameters)

---

**If you continue to experience issues, please:**
1. Collect debug information using the tools above
2. Check the log files for specific error messages
3. Try the recovery procedures
4. Create a detailed bug report with system information and error logs 