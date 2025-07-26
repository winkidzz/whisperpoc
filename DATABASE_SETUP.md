# WhisperPOC Database Setup Guide

## 🗄️ **Overview**

This guide provides step-by-step instructions for setting up PostgreSQL and initializing the database for the WhisperPOC audio storage system.

## 🚀 **Quick Setup (5 Minutes)**

### **Automated Setup**
```bash
# 1. Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# 2. Create database
createdb audio_storage_db

# 3. Initialize tables (automatic on first run)
python audio_storage_system.py --init-only
```

## 📋 **Prerequisites**

### **System Requirements**
- **Operating System**: macOS, Ubuntu/Debian, Windows
- **Python**: 3.8 or higher
- **PostgreSQL**: 12 or higher
- **Memory**: 2GB+ RAM (4GB+ recommended)
- **Storage**: 1GB+ free space

### **Check Current Setup**
```bash
# Check Python version
python3 --version

# Check if PostgreSQL is installed
psql --version

# Check if database exists
psql -l | grep audio_storage_db
```

## 🛠️ **PostgreSQL Installation**

### **macOS (Homebrew)**
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PostgreSQL
brew install postgresql

# Start PostgreSQL service
brew services start postgresql

# Verify installation
psql --version
```

### **Ubuntu/Debian**
```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify installation
psql --version
```

### **Windows**
1. Download PostgreSQL from: https://www.postgresql.org/download/windows/
2. Run the installer
3. Follow the installation wizard
4. Add PostgreSQL to your PATH environment variable

## 🗃️ **Database Creation**

### **Method 1: Using createdb (Recommended)**
```bash
# Create database
createdb audio_storage_db

# Verify creation
psql -l | grep audio_storage_db
```

### **Method 2: Using psql**
```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE audio_storage_db;

# Verify creation
\l

# Exit psql
\q
```

### **Method 3: Using pgAdmin (GUI)**
1. Open pgAdmin
2. Right-click on "Databases"
3. Select "Create" > "Database"
4. Enter "audio_storage_db" as the name
5. Click "Save"

## 🔧 **Database Configuration**

### **Environment Variables**
Create a `.env` file in your project directory:
```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=audio_storage_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=

# Optional: Connection pool settings
DB_CONNECTION_TIMEOUT=30
DB_MAX_CONNECTIONS=20
```

### **Load Environment Variables**
```bash
# Load environment variables
export $(cat .env | xargs)

# Or source the file
source .env
```

### **Test Database Connection**
```bash
# Test connection
psql -h localhost -p 5432 -U postgres -d audio_storage_db -c "SELECT version();"

# Test with Python
python -c "
from audio_storage_system import AudioStorageSystem
storage = AudioStorageSystem(storage_backend='postgres')
print('Database connection successful!')
"
```

## 🏗️ **Table Schema**

### **Automatic Table Creation**
The system automatically creates tables on first run. Here's the schema:

#### **audio_files Table**
```sql
CREATE TABLE audio_files (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **transcripts Table**
```sql
CREATE TABLE transcripts (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    transcription TEXT NOT NULL,
    confidence REAL,
    language VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **embeddings Table**
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL,
    embedding_data BYTEA NOT NULL,
    embedding_dimension INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### **metadata Table**
```sql
CREATE TABLE metadata (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Manual Table Creation**
If you prefer to create tables manually:

```bash
# Connect to database
psql -d audio_storage_db

# Run the schema creation script
\i schema.sql

# Or create tables individually
CREATE TABLE audio_files (
    id SERIAL PRIMARY KEY,
    file_hash VARCHAR(255) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Continue with other tables...
```

## 📊 **Indexes and Performance**

### **Create Performance Indexes**
```sql
-- Audio files indexes
CREATE INDEX idx_audio_files_created_at ON audio_files(created_at);
CREATE INDEX idx_audio_files_file_hash ON audio_files(file_hash);
CREATE INDEX idx_audio_files_duration ON audio_files(duration);

-- Transcripts indexes
CREATE INDEX idx_transcripts_confidence ON transcripts(confidence);
CREATE INDEX idx_transcripts_language ON transcripts(language);
CREATE INDEX idx_transcripts_audio_file_id ON transcripts(audio_file_id);

-- Embeddings indexes
CREATE INDEX idx_embeddings_type ON embeddings(embedding_type);
CREATE INDEX idx_embeddings_audio_file_id ON embeddings(audio_file_id);
CREATE INDEX idx_embeddings_transcript_id ON embeddings(transcript_id);

-- Metadata indexes
CREATE INDEX idx_metadata_key_value ON metadata(key, value);
CREATE INDEX idx_metadata_audio_file_id ON metadata(audio_file_id);
```

### **Analyze Tables**
```sql
-- Update table statistics for query optimization
ANALYZE audio_files;
ANALYZE transcripts;
ANALYZE embeddings;
ANALYZE metadata;
```

## 🔒 **Security Configuration**

### **User Permissions**
```sql
-- Create dedicated user (optional)
CREATE USER whisperpoc_user WITH PASSWORD 'your_secure_password';

-- Grant permissions
GRANT CONNECT ON DATABASE audio_storage_db TO whisperpoc_user;
GRANT USAGE ON SCHEMA public TO whisperpoc_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO whisperpoc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO whisperpoc_user;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO whisperpoc_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO whisperpoc_user;
```

### **Connection Security**
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/12/main/postgresql.conf

# Add/modify these settings:
listen_addresses = 'localhost'
port = 5432
max_connections = 100
shared_buffers = 128MB
effective_cache_size = 512MB
```

### **Authentication Configuration**
```bash
# Edit pg_hba.conf
sudo nano /etc/postgresql/12/main/pg_hba.conf

# Add these lines for local connections:
local   audio_storage_db    postgres                                peer
host    audio_storage_db    postgres        127.0.0.1/32            md5
host    audio_storage_db    postgres        ::1/128                 md5
```

## 📈 **Performance Tuning**

### **PostgreSQL Configuration**
```sql
-- Check current settings
SHOW shared_buffers;
SHOW effective_cache_size;
SHOW work_mem;
SHOW maintenance_work_mem;

-- Optimize for audio storage workload
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### **Connection Pooling**
```bash
# Install pgBouncer (optional)
sudo apt install pgbouncer

# Configure pgBouncer
sudo nano /etc/pgbouncer/pgbouncer.ini

[databases]
audio_storage_db = host=localhost port=5432 dbname=audio_storage_db

[pgbouncer]
listen_port = 6432
listen_addr = localhost
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

## 🔍 **Monitoring and Maintenance**

### **Database Health Check**
```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### **Backup and Recovery**
```bash
# Create backup
pg_dump audio_storage_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Create compressed backup
pg_dump audio_storage_db | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Restore from backup
psql audio_storage_db < backup_20240101_120000.sql

# Restore from compressed backup
gunzip -c backup_20240101_120000.sql.gz | psql audio_storage_db
```

### **Maintenance Tasks**
```sql
-- Vacuum tables (clean up dead tuples)
VACUUM ANALYZE audio_files;
VACUUM ANALYZE transcripts;
VACUUM ANALYZE embeddings;
VACUUM ANALYZE metadata;

-- Reindex tables (rebuild indexes)
REINDEX TABLE audio_files;
REINDEX TABLE transcripts;
REINDEX TABLE embeddings;
REINDEX TABLE metadata;
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Connection Refused**
```bash
# Check PostgreSQL status
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql      # Ubuntu

# Start PostgreSQL if stopped
brew services start postgresql        # macOS
sudo systemctl start postgresql       # Ubuntu
```

#### **Permission Denied**
```bash
# Grant permissions
psql -U postgres -d audio_storage_db
GRANT ALL PRIVILEGES ON DATABASE audio_storage_db TO your_username;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_username;
\q
```

#### **Database Does Not Exist**
```bash
# Create database
createdb audio_storage_db

# Or using psql
psql -U postgres
CREATE DATABASE audio_storage_db;
\q
```

#### **Port Already in Use**
```bash
# Check what's using port 5432
lsof -i :5432

# Kill process if needed
sudo kill -9 <PID>

# Or change PostgreSQL port
sudo nano /etc/postgresql/12/main/postgresql.conf
# Change port = 5433
```

### **Debug Connection Issues**
```bash
# Test connection with verbose output
psql -h localhost -p 5432 -U postgres -d audio_storage_db -v ON_ERROR_STOP=1

# Check PostgreSQL logs
tail -f /var/log/postgresql/postgresql-12-main.log  # Ubuntu
tail -f /usr/local/var/log/postgres.log            # macOS
```

## 📋 **Verification Checklist**

### **Pre-Setup Checklist**
- [ ] PostgreSQL installed and running
- [ ] Database created
- [ ] Environment variables configured
- [ ] Python dependencies installed
- [ ] Audio system dependencies installed

### **Post-Setup Verification**
```bash
# 1. Test database connection
python -c "from audio_storage_system import AudioStorageSystem; AudioStorageSystem(storage_backend='postgres')"

# 2. Test table creation
python audio_storage_system.py --init-only

# 3. Test audio recording
python interactive_recorder.py

# 4. Test embedding queries
python embedding_queries.py

# 5. Verify data persistence
psql -d audio_storage_db -c "SELECT COUNT(*) FROM audio_files;"
```

### **Performance Verification**
```bash
# Test query performance
psql -d audio_storage_db -c "EXPLAIN ANALYZE SELECT * FROM audio_files ORDER BY created_at DESC LIMIT 10;"

# Check index usage
psql -d audio_storage_db -c "SELECT * FROM pg_stat_user_indexes WHERE idx_scan > 0;"
```

## 🔮 **Advanced Configuration**

### **Replication Setup (Optional)**
```bash
# Primary server configuration
sudo nano /etc/postgresql/12/main/postgresql.conf

wal_level = replica
max_wal_senders = 3
max_replication_slots = 3

# Standby server configuration
sudo nano /etc/postgresql/12/main/postgresql.conf

hot_standby = on
```

### **Partitioning (For Large Datasets)**
```sql
-- Create partitioned table for large datasets
CREATE TABLE audio_files_partitioned (
    id SERIAL,
    file_hash VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration REAL,
    sample_rate INTEGER,
    channels INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

-- Create partitions by month
CREATE TABLE audio_files_2024_01 PARTITION OF audio_files_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE audio_files_2024_02 PARTITION OF audio_files_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

---

**This guide ensures your PostgreSQL database is properly configured for optimal performance with the WhisperPOC audio storage system.** 