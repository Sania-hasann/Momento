"""
Configuration settings for the RAG pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for database and model settings."""
    
    # Database settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'rag_database')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # Vector settings
    VECTOR_DIMENSION = int(os.getenv('VECTOR_DIMENSION', 384))  # all-MiniLM-L6-v2 dimension
    
    # Model settings
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Table names
    DOCUMENTS_TABLE = 'data_documents'
    
    # LlamaIndex settings - OPTIMIZED: Smaller chunks for better granularity and diversity
    CHUNK_SIZE = 256  # Reduced from 512 for better granularity and diversity
    CHUNK_OVERLAP = 25  # Reduced from 50 for less redundancy
    CHUNK_SEPARATOR = "\n\n"  # Use paragraph breaks for better chunking
    
    # Search settings
    DEFAULT_K = 5
    DEFAULT_HYBRID_WEIGHT = 0.5
    MAX_SEARCH_CANDIDATES = 20  # For better diversity
    
    # BM25 settings
    BM25_K1 = 1.2  # BM25 k1 parameter (controls term frequency saturation)
    BM25_B = 0.75  # BM25 b parameter (controls length normalization)
    
    # Vector search settings
    SIMILARITY_TOP_K = 10  # Default similarity top-k for retriever
    VECTOR_SEARCH_MULTIPLIER = 2  # Multiplier for vector search candidates (k * multiplier)
    
    # Hybrid search settings
    HYBRID_SEARCH_MULTIPLIER = 3  # Multiplier for hybrid search candidates (k * multiplier)
    DEFAULT_VECTOR_WEIGHT = 0.7  # Default vector weight for hybrid search
    DEFAULT_FUSION_METHOD = "wlc"  # Default fusion method ("wlc", "rrf", "adaptive")
    
    # Query-aware weighting settings
    KEYWORD_HEAVY_WEIGHT = 0.4  # Vector weight for keyword-heavy queries
    SEMANTIC_HEAVY_WEIGHT = 0.8  # Vector weight for semantic-heavy queries
    BALANCED_WEIGHT = 0.7  # Vector weight for balanced queries
    
    # Score normalization settings
    SIGMOID_SCALING_FACTOR = 5  # Scaling factor for sigmoid transformation
    DEFAULT_NORMALIZED_SCORE = 0.5  # Default score when all scores are equal
    
    # Deduplication settings
    SIMILARITY_THRESHOLD = 0.8  # Default similarity threshold for deduplication
    HYBRID_SIMILARITY_THRESHOLD = 0.9  # Similarity threshold for hybrid search deduplication
    CONTENT_PREVIEW_LENGTH = 100  # Length of content preview for deduplication
    CONTENT_HASH_LENGTH = 500  # Length of content for hash generation
    
    # Full-text search settings
    MIN_KEYWORD_LENGTH = 3  # Minimum keyword length for search
    EXACT_MATCH_BONUS = 0.3  # Bonus multiplier for exact matches
    FULL_TEXT_SEARCH_MULTIPLIER = 2  # Multiplier for full-text search candidates
    
    # RRF (Reciprocal Rank Fusion) settings
    RRF_K = 60  # RRF parameter (typically 60)
    
    # Content processing settings
    CONTENT_MATCH_LENGTH = 100  # Length of content for database matching
    
    # Performance settings
    BATCH_SIZE = 10  # For batch operations
    CONNECTION_TIMEOUT = 30  # Database connection timeout
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get the PostgreSQL connection URL."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
