"""
WhisperPOC Configuration Settings

This file contains all configuration settings for the WhisperPOC project.
Settings can be overridden by environment variables or command-line arguments.
"""

import os
from typing import Dict, Any

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Default PostgreSQL configuration
DEFAULT_POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'audio_storage_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', ''),
}

# Database connection timeout (seconds)
DB_CONNECTION_TIMEOUT = int(os.getenv('DB_CONNECTION_TIMEOUT', 30))

# Maximum database connections
DB_MAX_CONNECTIONS = int(os.getenv('DB_MAX_CONNECTIONS', 20))

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

# Default audio recording settings
DEFAULT_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', 16000))
DEFAULT_CHANNELS = int(os.getenv('AUDIO_CHANNELS', 1))
DEFAULT_CHUNK_SIZE = int(os.getenv('AUDIO_CHUNK_SIZE', 1024))
DEFAULT_RECORDING_DURATION = int(os.getenv('RECORDING_DURATION', 10))

# Audio file formats
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.aac']

# Audio storage directory
AUDIO_STORAGE_DIR = os.getenv('AUDIO_STORAGE_DIR', 'audio_storage')

# Maximum audio file size (MB)
MAX_AUDIO_FILE_SIZE = int(os.getenv('MAX_AUDIO_FILE_SIZE', 100))

# =============================================================================
# WHISPER CONFIGURATION
# =============================================================================

# Default Whisper model
DEFAULT_WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')

# Available Whisper models
AVAILABLE_WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Whisper device (cpu/cuda)
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')

# Whisper language (None for auto-detection)
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', None)

# Whisper confidence threshold
WHISPER_CONFIDENCE_THRESHOLD = float(os.getenv('WHISPER_CONFIDENCE_THRESHOLD', 0.0))

# Whisper cache directory
WHISPER_CACHE_DIR = os.getenv('WHISPER_CACHE_DIR', os.path.expanduser('~/.cache/whisper'))

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Default embedding model
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')

# Embedding dimensions
EMBEDDING_DIMENSIONS = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'sentence-transformers': 768,
}

# Default embedding dimension
DEFAULT_EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 1536))

# Similarity search settings
SIMILARITY_SEARCH_LIMIT = int(os.getenv('SIMILARITY_SEARCH_LIMIT', 10))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))

# =============================================================================
# VOICE-TO-LLM CONFIGURATION
# =============================================================================

# API Keys (load from environment variables)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# Default LLM provider
DEFAULT_LLM_PROVIDER = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')

# LLM model settings
LLM_MODELS = {
    'openai': {
        'default': 'gpt-4o',
        'available': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
    },
    'anthropic': {
        'default': 'claude-3-5-sonnet-20241022',
        'available': ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229']
    },
    'google': {
        'default': 'gemini-1.5-pro',
        'available': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
    }
}

# Local LLM settings
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LOCALAI_BASE_URL = os.getenv('LOCALAI_BASE_URL', 'http://localhost:8080')

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Debug mode
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Logging level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Log file
LOG_FILE = os.getenv('LOG_FILE', 'whisperpoc.log')

# Application name
APP_NAME = 'WhisperPOC'

# Application version
APP_VERSION = '1.0.0'

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Batch processing size
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))

# Thread pool size
THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', 4))

# Memory limit (MB)
MEMORY_LIMIT = int(os.getenv('MEMORY_LIMIT', 1024))

# Cache settings
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

# API rate limiting
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # 1 hour

# File upload security
ALLOWED_FILE_EXTENSIONS = SUPPORTED_AUDIO_FORMATS
MAX_FILE_SIZE = MAX_AUDIO_FILE_SIZE * 1024 * 1024  # Convert to bytes

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

# Export formats
SUPPORTED_EXPORT_FORMATS = ['csv', 'json', 'npz', 'sql']

# Export directory
EXPORT_DIR = os.getenv('EXPORT_DIR', 'exports')

# Export file naming
EXPORT_FILE_PREFIX = os.getenv('EXPORT_FILE_PREFIX', 'whisperpoc_export')

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Input validation
VALIDATE_AUDIO_FILES = os.getenv('VALIDATE_AUDIO_FILES', 'true').lower() == 'true'
VALIDATE_TRANSCRIPTS = os.getenv('VALIDATE_TRANSCRIPTS', 'true').lower() == 'true'
VALIDATE_EMBEDDINGS = os.getenv('VALIDATE_EMBEDDINGS', 'true').lower() == 'true'

# Data integrity checks
CHECK_DATA_INTEGRITY = os.getenv('CHECK_DATA_INTEGRITY', 'true').lower() == 'true'

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_postgres_config() -> Dict[str, Any]:
    """Get PostgreSQL configuration with environment variable overrides."""
    config = DEFAULT_POSTGRES_CONFIG.copy()
    
    # Override with environment variables if present
    if os.getenv('POSTGRES_HOST'):
        config['host'] = os.getenv('POSTGRES_HOST')
    if os.getenv('POSTGRES_PORT'):
        config['port'] = int(os.getenv('POSTGRES_PORT'))
    if os.getenv('POSTGRES_DB'):
        config['database'] = os.getenv('POSTGRES_DB')
    if os.getenv('POSTGRES_USER'):
        config['user'] = os.getenv('POSTGRES_USER')
    if os.getenv('POSTGRES_PASSWORD'):
        config['password'] = os.getenv('POSTGRES_PASSWORD')
    
    return config

def get_whisper_config() -> Dict[str, Any]:
    """Get Whisper configuration."""
    return {
        'model': os.getenv('WHISPER_MODEL', DEFAULT_WHISPER_MODEL),
        'device': WHISPER_DEVICE,
        'language': WHISPER_LANGUAGE,
        'confidence_threshold': WHISPER_CONFIDENCE_THRESHOLD,
        'cache_dir': WHISPER_CACHE_DIR
    }

def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration."""
    return {
        'model': os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL),
        'dimension': DEFAULT_EMBEDDING_DIMENSION,
        'similarity_limit': SIMILARITY_SEARCH_LIMIT,
        'similarity_threshold': SIMILARITY_THRESHOLD
    }

def get_audio_config() -> Dict[str, Any]:
    """Get audio configuration."""
    return {
        'sample_rate': DEFAULT_SAMPLE_RATE,
        'channels': DEFAULT_CHANNELS,
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'duration': DEFAULT_RECORDING_DURATION,
        'storage_dir': AUDIO_STORAGE_DIR,
        'max_file_size': MAX_AUDIO_FILE_SIZE,
        'supported_formats': SUPPORTED_AUDIO_FORMATS
    }

def validate_config() -> bool:
    """Validate configuration settings."""
    errors = []
    
    # Check PostgreSQL configuration
    postgres_config = get_postgres_config()
    if not postgres_config['database']:
        errors.append("PostgreSQL database name is required")
    
    # Check Whisper model
    if DEFAULT_WHISPER_MODEL not in AVAILABLE_WHISPER_MODELS:
        errors.append(f"Invalid Whisper model: {DEFAULT_WHISPER_MODEL}")
    
    # Check embedding dimension
    if DEFAULT_EMBEDDING_DIMENSION <= 0:
        errors.append("Embedding dimension must be positive")
    
    # Check audio settings
    if DEFAULT_SAMPLE_RATE <= 0:
        errors.append("Sample rate must be positive")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def print_config_summary():
    """Print a summary of current configuration."""
    print("=" * 60)
    print("WhisperPOC Configuration Summary")
    print("=" * 60)
    
    print(f"Database: {get_postgres_config()['database']} on {get_postgres_config()['host']}:{get_postgres_config()['port']}")
    print(f"Whisper Model: {DEFAULT_WHISPER_MODEL}")
    print(f"Audio Settings: {DEFAULT_SAMPLE_RATE}Hz, {DEFAULT_CHANNELS} channel(s), {DEFAULT_RECORDING_DURATION}s duration")
    print(f"Embedding Model: {DEFAULT_EMBEDDING_MODEL} ({DEFAULT_EMBEDDING_DIMENSION} dimensions)")
    print(f"Debug Mode: {DEBUG}")
    print(f"Log Level: {LOG_LEVEL}")
    print("=" * 60)

# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Development settings
if os.getenv('ENVIRONMENT') == 'development':
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False

# Production settings
elif os.getenv('ENVIRONMENT') == 'production':
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    CACHE_ENABLED = True
    RATE_LIMIT_ENABLED = True

# Test settings
elif os.getenv('ENVIRONMENT') == 'test':
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False
    DEFAULT_POSTGRES_CONFIG['database'] = 'test_audio_storage_db'

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Validate configuration on import
    if not validate_config():
        raise ValueError("Invalid configuration detected")
    
    # Print configuration summary
    print_config_summary() 