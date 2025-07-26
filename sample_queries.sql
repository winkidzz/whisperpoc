-- =============================================================================
-- WhisperPOC Sample SQL Queries
-- =============================================================================
-- This file contains sample SQL queries for the WhisperPOC audio storage system
-- Use these queries to explore and analyze your audio data

-- =============================================================================
-- BASIC QUERIES
-- =============================================================================

-- 1. View all audio files with basic information
SELECT 
    id,
    file_path,
    file_size,
    duration,
    sample_rate,
    channels,
    created_at
FROM audio_files
ORDER BY created_at DESC;

-- 2. Count total audio files
SELECT COUNT(*) as total_audio_files FROM audio_files;

-- 3. Get audio files by date range
SELECT 
    file_path,
    duration,
    created_at
FROM audio_files
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;

-- =============================================================================
-- TRANSCRIPTION QUERIES
-- =============================================================================

-- 4. View all transcriptions with confidence scores
SELECT 
    af.file_path,
    t.transcription,
    t.confidence,
    t.language,
    t.created_at
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
ORDER BY t.confidence DESC;

-- 5. Find high-confidence transcriptions (>90%)
SELECT 
    af.file_path,
    t.transcription,
    t.confidence
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
WHERE t.confidence > 0.9
ORDER BY t.confidence DESC;

-- 6. Find low-confidence transcriptions (<50%)
SELECT 
    af.file_path,
    t.transcription,
    t.confidence
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
WHERE t.confidence < 0.5
ORDER BY t.confidence ASC;

-- 7. Search transcriptions by keyword
SELECT 
    af.file_path,
    t.transcription,
    t.confidence
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
WHERE t.transcription ILIKE '%meeting%'
ORDER BY t.confidence DESC;

-- 8. Get transcription statistics
SELECT 
    COUNT(*) as total_transcriptions,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence,
    COUNT(CASE WHEN confidence > 0.9 THEN 1 END) as high_confidence_count
FROM transcripts;

-- =============================================================================
-- METADATA QUERIES
-- =============================================================================

-- 9. View all metadata for audio files
SELECT 
    af.file_path,
    m.key,
    m.value,
    m.created_at
FROM audio_files af
JOIN metadata m ON af.id = m.audio_file_id
ORDER BY af.created_at DESC, m.key;

-- 10. Find audio files by user
SELECT 
    af.file_path,
    t.transcription,
    m.value as user
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'user'
ORDER BY af.created_at DESC;

-- 11. Find audio files by session
SELECT 
    af.file_path,
    t.transcription,
    m.value as session
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'session'
ORDER BY af.created_at DESC;

-- 12. Find high-priority recordings
SELECT 
    af.file_path,
    t.transcription,
    m.value as priority
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'priority' AND m.value = 'high'
ORDER BY af.created_at DESC;

-- 13. Get metadata statistics
SELECT 
    key,
    COUNT(*) as count,
    COUNT(DISTINCT value) as unique_values
FROM metadata
GROUP BY key
ORDER BY count DESC;

-- =============================================================================
-- EMBEDDING QUERIES
-- =============================================================================

-- 14. View all embeddings with metadata
SELECT 
    e.id,
    e.embedding_type,
    e.embedding_dimension,
    t.transcription,
    e.created_at
FROM embeddings e
JOIN transcripts t ON e.transcript_id = t.id
ORDER BY e.created_at DESC;

-- 15. Count embeddings by type
SELECT 
    embedding_type,
    COUNT(*) as count,
    AVG(embedding_dimension) as avg_dimension
FROM embeddings
GROUP BY embedding_type
ORDER BY count DESC;

-- 16. Find embeddings for specific audio files
SELECT 
    af.file_path,
    e.embedding_type,
    e.embedding_dimension,
    t.transcription
FROM embeddings e
JOIN audio_files af ON e.audio_file_id = af.id
JOIN transcripts t ON e.transcript_id = t.id
ORDER BY af.created_at DESC;

-- =============================================================================
-- ADVANCED QUERIES
-- =============================================================================

-- 17. Get complete audio file information
SELECT 
    af.id,
    af.file_path,
    af.file_size,
    af.duration,
    af.sample_rate,
    af.channels,
    t.transcription,
    t.confidence,
    t.language,
    e.embedding_type,
    e.embedding_dimension,
    af.created_at
FROM audio_files af
LEFT JOIN transcripts t ON af.id = t.audio_file_id
LEFT JOIN embeddings e ON af.id = e.audio_file_id
ORDER BY af.created_at DESC;

-- 18. Find audio files with missing transcriptions
SELECT 
    af.id,
    af.file_path,
    af.duration,
    af.created_at
FROM audio_files af
LEFT JOIN transcripts t ON af.id = t.audio_file_id
WHERE t.id IS NULL
ORDER BY af.created_at DESC;

-- 19. Find audio files with missing embeddings
SELECT 
    af.id,
    af.file_path,
    t.transcription,
    af.created_at
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
LEFT JOIN embeddings e ON af.id = e.audio_file_id
WHERE e.id IS NULL
ORDER BY af.created_at DESC;

-- 20. Get audio files by duration range
SELECT 
    file_path,
    duration,
    created_at
FROM audio_files
WHERE duration BETWEEN 5 AND 30  -- 5-30 seconds
ORDER BY duration DESC;

-- 21. Get audio files by file size range
SELECT 
    file_path,
    file_size,
    duration,
    created_at
FROM audio_files
WHERE file_size BETWEEN 100000 AND 1000000  -- 100KB to 1MB
ORDER BY file_size DESC;

-- =============================================================================
-- AGGREGATION QUERIES
-- =============================================================================

-- 22. Daily audio recording statistics
SELECT 
    DATE(created_at) as recording_date,
    COUNT(*) as recordings_count,
    AVG(duration) as avg_duration,
    SUM(file_size) as total_size_bytes,
    AVG(t.confidence) as avg_confidence
FROM audio_files af
LEFT JOIN transcripts t ON af.id = t.audio_file_id
GROUP BY DATE(created_at)
ORDER BY recording_date DESC;

-- 23. User activity statistics
SELECT 
    m.value as user,
    COUNT(*) as recordings_count,
    AVG(af.duration) as avg_duration,
    AVG(t.confidence) as avg_confidence
FROM audio_files af
JOIN metadata m ON af.id = m.audio_file_id
JOIN transcripts t ON af.id = t.audio_file_id
WHERE m.key = 'user'
GROUP BY m.value
ORDER BY recordings_count DESC;

-- 24. Session statistics
SELECT 
    m.value as session,
    COUNT(*) as recordings_count,
    AVG(af.duration) as avg_duration,
    MIN(af.created_at) as session_start,
    MAX(af.created_at) as session_end
FROM audio_files af
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'session'
GROUP BY m.value
ORDER BY recordings_count DESC;

-- 25. Language distribution
SELECT 
    language,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence
FROM transcripts
WHERE language IS NOT NULL
GROUP BY language
ORDER BY count DESC;

-- =============================================================================
-- PERFORMANCE QUERIES
-- =============================================================================

-- 26. Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 27. Index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 28. Table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_live_tup as live_tuples,
    n_dead_tup as dead_tuples
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

-- =============================================================================
-- MAINTENANCE QUERIES
-- =============================================================================

-- 29. Find orphaned records
SELECT 'orphaned_transcripts' as type, COUNT(*) as count
FROM transcripts t
LEFT JOIN audio_files af ON t.audio_file_id = af.id
WHERE af.id IS NULL
UNION ALL
SELECT 'orphaned_embeddings' as type, COUNT(*) as count
FROM embeddings e
LEFT JOIN audio_files af ON e.audio_file_id = af.id
WHERE af.id IS NULL
UNION ALL
SELECT 'orphaned_metadata' as type, COUNT(*) as count
FROM metadata m
LEFT JOIN audio_files af ON m.audio_file_id = af.id
WHERE af.id IS NULL;

-- 30. Clean up orphaned records (use with caution!)
-- DELETE FROM transcripts WHERE audio_file_id NOT IN (SELECT id FROM audio_files);
-- DELETE FROM embeddings WHERE audio_file_id NOT IN (SELECT id FROM audio_files);
-- DELETE FROM metadata WHERE audio_file_id NOT IN (SELECT id FROM audio_files);

-- 31. Update table statistics
-- ANALYZE audio_files;
-- ANALYZE transcripts;
-- ANALYZE embeddings;
-- ANALYZE metadata;

-- =============================================================================
-- EXPORT QUERIES
-- =============================================================================

-- 32. Export all data for CSV
SELECT 
    af.id,
    af.file_path,
    af.file_size,
    af.duration,
    af.sample_rate,
    af.channels,
    t.transcription,
    t.confidence,
    t.language,
    e.embedding_type,
    e.embedding_dimension,
    af.created_at
FROM audio_files af
LEFT JOIN transcripts t ON af.id = t.audio_file_id
LEFT JOIN embeddings e ON af.id = e.audio_file_id
ORDER BY af.created_at DESC;

-- 33. Export metadata for analysis
SELECT 
    af.id,
    af.file_path,
    m.key,
    m.value,
    af.created_at
FROM audio_files af
JOIN metadata m ON af.id = m.audio_file_id
ORDER BY af.created_at DESC, m.key;

-- =============================================================================
-- USAGE EXAMPLES
-- =============================================================================

-- Example 1: Find all recordings from today
SELECT file_path, duration, created_at
FROM audio_files
WHERE DATE(created_at) = CURRENT_DATE;

-- Example 2: Find recordings longer than 1 minute
SELECT file_path, duration
FROM audio_files
WHERE duration > 60
ORDER BY duration DESC;

-- Example 3: Find recordings with specific tags
SELECT af.file_path, t.transcription, m.value as tag
FROM audio_files af
JOIN transcripts t ON af.id = t.audio_file_id
JOIN metadata m ON af.id = m.audio_file_id
WHERE m.key = 'tags' AND m.value ILIKE '%important%';

-- Example 4: Get the most recent recording for each user
SELECT DISTINCT ON (m.value) 
    m.value as user,
    af.file_path,
    t.transcription,
    af.created_at
FROM audio_files af
JOIN metadata m ON af.id = m.audio_file_id
JOIN transcripts t ON af.id = t.audio_file_id
WHERE m.key = 'user'
ORDER BY m.value, af.created_at DESC;

-- Example 5: Find duplicate file hashes
SELECT file_hash, COUNT(*) as count
FROM audio_files
GROUP BY file_hash
HAVING COUNT(*) > 1; 