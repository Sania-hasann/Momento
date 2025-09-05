"""
Document ingestion module for the RAG pipeline using LlamaIndex.
Uses custom splitters for better handling of markdown and Unicode content.
"""
import os
import logging
from typing import List, Dict, Any, Union
from llama_index.legacy import VectorStoreIndex, Document, ServiceContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.node_parser import SentenceSplitter
from custom_node_parser import MarkdownTokenTextSplitter, UnicodeSafeTokenTextSplitter
from config import Config

logger = logging.getLogger(__name__)

class DocumentIngestion:
    """
    Handles document ingestion using LlamaIndex with PostgreSQL vector store.
    
    This class processes raw text documents or file paths, generates embeddings
    using the all-MiniLM-L6-v2 model, and stores them in PostgreSQL using
    LlamaIndex's PGVectorStore for efficient similarity search.
    Uses custom splitters for better handling of markdown and Unicode content.
    """
    
    def __init__(self):
        """Initialize the document ingestion pipeline with LlamaIndex."""
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize custom splitters
        self.markdown_splitter = MarkdownTokenTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.unicode_splitter = UnicodeSafeTokenTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.default_node_parser = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator=Config.CHUNK_SEPARATOR
        )
        
        # Initialize service context with MarkdownTokenTextSplitter as default
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embedding_model,
            node_parser=self.markdown_splitter,  # Use MarkdownTokenTextSplitter as default
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
        
        logger.info(f"Initialized DocumentIngestion with custom splitters and model: {Config.EMBEDDING_MODEL}")
    
    def _get_appropriate_splitter(self, text: str, file_path: str = None) -> Union[
        MarkdownTokenTextSplitter, UnicodeSafeTokenTextSplitter, SentenceSplitter
    ]:
        """
        Select the appropriate splitter based on content type.
        
        Args:
            text: The text content to analyze
            file_path: Optional file path to help determine content type
            
        Returns:
            The appropriate splitter instance
        """
        # Check if it's a markdown file
        if file_path and file_path.endswith('.md'):
            return self.markdown_splitter
        
        # Check if text contains markdown syntax
        if any(marker in text for marker in ['# ', '## ', '### ', '|', '```', '**', '*']):
            return self.markdown_splitter
        
        # Check for Unicode characters that might need special handling
        if any(ord(char) > 127 for char in text[:1000]):  # Check first 1000 chars
            return self.unicode_splitter
        
        # Default to markdown splitter for better general handling
        return self.markdown_splitter
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None, file_path: str = None) -> str:
        """
        Ingest a single text document using LlamaIndex with appropriate custom splitter.
        
        Args:
            text: The text content to ingest
            metadata: Optional metadata dictionary to store with the document
            file_path: Optional file path to help determine content type
            
        Returns:
            str: The document ID
        """
        try:
            import uuid
            
            # Select appropriate splitter
            splitter = self._get_appropriate_splitter(text, file_path)
            splitter_name = splitter.__class__.__name__
            
            # Generate a unique document ID
            doc_id = str(uuid.uuid4())
            
            # Add splitter information to metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata['splitter_used'] = splitter_name
            
            # Create LlamaIndex Document with the generated ID
            document = Document(
                text=text,
                metadata=enhanced_metadata,
                id_=doc_id
            )
            
            # Temporarily update service context with selected splitter
            original_node_parser = self.service_context.node_parser
            self.service_context.node_parser = splitter
            
            # Insert document into index
            self.index.insert(document)
            
            # Restore original node parser
            self.service_context.node_parser = original_node_parser
            
            logger.info(f"Ingested text document with ID: {doc_id} using {splitter_name}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error ingesting text document: {e}")
            raise
    
    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a document from a file path using LlamaIndex with appropriate custom splitter.
        
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
            
            return self.ingest_text(content, file_metadata, file_path)
            
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
    
    def ingest_markdown_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """
        Recursively ingest all markdown files from a directory using custom markdown splitter.
        
        Args:
            directory_path: Path to the directory containing markdown files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List[str]: List of document IDs that were inserted
        """
        import glob
        
        doc_ids = []
        
        # Find all markdown files
        if recursive:
            pattern = os.path.join(directory_path, "**", "*.md")
        else:
            pattern = os.path.join(directory_path, "*.md")
        
        markdown_files = glob.glob(pattern, recursive=recursive)
        
        if not markdown_files:
            logger.warning(f"No markdown files found in {directory_path}")
            return doc_ids
        
        logger.info(f"Starting markdown ingestion from directory: {directory_path} using custom markdown splitter")
        logger.info(f"Found {len(markdown_files)} markdown files to ingest")
        
        for file_path in markdown_files:
            try:
                # Add directory metadata
                relative_path = os.path.relpath(file_path, directory_path)
                directory = os.path.dirname(relative_path)
                
                metadata = {
                    'file_name': os.path.basename(file_path),
                    'directory': directory,
                    'file_type': 'markdown',
                    'splitter_used': 'MarkdownTokenTextSplitter'  # Track which splitter was used
                }
                
                doc_id = self.ingest_file(file_path, metadata)
                doc_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error ingesting file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully ingested {len(doc_ids)} markdown files")
        return doc_ids
    
    def get_splitter_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics about which splitters were used for ingested documents.
        
        Returns:
            Dict[str, int]: Dictionary with splitter names as keys and usage counts as values
        """
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        metadata_->>'splitter_used' as splitter,
                        COUNT(*) as count
                    FROM {Config.DOCUMENTS_TABLE}
                    WHERE metadata_->>'splitter_used' IS NOT NULL
                    GROUP BY metadata_->>'splitter_used'
                    ORDER BY count DESC
                """)
                results = cursor.fetchall()
                
                stats = {row[0]: row[1] for row in results}
            
            conn.close()
            logger.info(f"Splitter usage stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting splitter usage stats: {e}")
            return {}
    
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
