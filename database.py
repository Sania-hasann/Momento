"""
Database connection and schema management for the RAG pipeline.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional
import logging
from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connection and schema."""

    def __init__(self):
        """Initialize the database manager."""
        self.connection = None
        self._ensure_pgvector_extension()
        self._create_documents_table()

    def get_connection(self):
        """Get a database connection."""
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(
                    host=Config.DB_HOST,
                    port=Config.DB_PORT,
                    database=Config.DB_NAME,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD,
                    cursor_factory=RealDictCursor,
                )
                logger.info("Database connection established successfully")
            except psycopg2.Error as e:
                logger.error(f"Error connecting to database: {e}")
                raise
        return self.connection

    def _ensure_pgvector_extension(self):
        """Ensure the pgvector extension is installed and enabled."""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("pgvector extension ensured")
        except psycopg2.Error as e:
            logger.error(f"Error ensuring pgvector extension: {e}")
            raise

    def _create_documents_table(self):
        """Create the documents table with vector support."""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {Config.DOCUMENTS_TABLE} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({Config.VECTOR_DIMENSION}),
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create indexes for better performance
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{Config.DOCUMENTS_TABLE}_embedding 
                    ON {Config.DOCUMENTS_TABLE} 
                    USING ivfflat (embedding vector_cosine_ops);
                """)

                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{Config.DOCUMENTS_TABLE}_content 
                    ON {Config.DOCUMENTS_TABLE} 
                    USING gin(to_tsvector('english', content));
                """)

                conn.commit()
                logger.info(
                    f"Documents table '{Config.DOCUMENTS_TABLE}' created successfully"
                )
        except psycopg2.Error as e:
            logger.error(f"Error creating documents table: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
