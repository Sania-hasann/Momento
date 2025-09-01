"""
Document ingestion module for the RAG pipeline using LlamaIndex.
"""
import os
import logging
from typing import List, Dict, Any, Union
from llama_index.legacy import VectorStoreIndex, Document, ServiceContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.node_parser import SentenceSplitter
from config import Config

logger = logging.getLogger(__name__)

class DocumentIngestion:
    """
    Handles document ingestion using LlamaIndex with PostgreSQL vector store.
    
    This class processes raw text documents or file paths, generates embeddings
    using the all-MiniLM-L6-v2 model, and stores them in PostgreSQL using
    LlamaIndex's PGVectorStore for efficient similarity search.
    """
    
    def __init__(self):
        """Initialize the document ingestion pipeline with LlamaIndex."""
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize node parser with improved chunking to reduce duplicates
        self.node_parser = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator=Config.CHUNK_SEPARATOR
        )
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embedding_model,
            node_parser=self.node_parser,
            llm=None  # We don't need an LLM for embeddings
        )
        
        # Initialize vector store
        self.vector_store = PGVectorStore.from_params(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            table_name=Config.DOCUMENTS_TABLE,
            embed_dim=Config.VECTOR_DIMENSION
        )
        
        # Initialize index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            service_context=self.service_context
        )
        
        logger.info(f"Initialized DocumentIngestion with LlamaIndex and model: {Config.EMBEDDING_MODEL}")
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a single text document using LlamaIndex.
        
        Args:
            text: The text content to ingest
            metadata: Optional metadata dictionary to store with the document
            
        Returns:
            str: The document ID
        """
        try:
            import uuid
            
            # Generate a unique document ID
            doc_id = str(uuid.uuid4())
            
            # Create LlamaIndex Document with the generated ID
            document = Document(
                text=text,
                metadata=metadata or {},
                id_=doc_id
            )
            
            # Insert document into index
            self.index.insert(document)
            
            logger.info(f"Ingested text document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error ingesting text document: {e}")
            raise
    
    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a document from a file path using LlamaIndex.
        
        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata dictionary to store with the document
            
        Returns:
            str: The document ID
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Add file metadata
            file_metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path)
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            return self.ingest_text(content, file_metadata)
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            raise
    
    def ingest_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """
        Ingest multiple documents using LlamaIndex.
        
        Args:
            documents: List of documents. Each document can be:
                - A string (text content)
                - A dictionary with 'text' and optional 'metadata' keys
                - A dictionary with 'file_path' and optional 'metadata' keys
                
        Returns:
            List[str]: List of document IDs that were inserted
        """
        doc_ids = []
        
        for doc in documents:
            try:
                if isinstance(doc, str):
                    # Plain text
                    doc_id = self.ingest_text(doc)
                elif isinstance(doc, dict):
                    if 'text' in doc:
                        # Text with metadata
                        doc_id = self.ingest_text(doc['text'], doc.get('metadata'))
                    elif 'file_path' in doc:
                        # File path with metadata
                        doc_id = self.ingest_file(doc['file_path'], doc.get('metadata'))
                    else:
                        raise ValueError("Document must have 'text' or 'file_path' key")
                else:
                    raise ValueError(f"Unsupported document type: {type(doc)}")
                
                doc_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error ingesting document: {e}")
                raise
        
        logger.info(f"Successfully ingested {len(doc_ids)} documents")
        return doc_ids
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            int: Number of documents
        """
        try:
            # Use psycopg2 to connect directly to the database
            import psycopg2
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {Config.DOCUMENTS_TABLE}")
                result = cursor.fetchone()
                count = result[0] if result else 0
            
            conn.close()
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0  # Return 0 instead of raising to avoid breaking the pipeline
    
    def close(self):
        """Close the vector store connection."""
        try:
            # The vector store will handle its own connection cleanup
            logger.info("DocumentIngestion connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
