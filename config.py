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
    DOCUMENTS_TABLE = 'documents'
    
    # LlamaIndex settings
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 20
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get the PostgreSQL connection URL."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"



config = Config()
