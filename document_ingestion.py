"""
Document ingestion module for the RAG pipeline using LlamaIndex.
"""
import os
import logging
from typing import List, Dict, Any, Union
from llama_index.legacy import VectorStoreIndex, ServiceContext
from llama_index.core.schema import Document
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.node_parser import SentenceSplitter
from config import Config
from custom_node_parser import MarkdownTokenTextSplitter, UnicodeSafeTokenTextSplitter

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
        
        # Initialize default node parser for general text
        self.default_node_parser = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator="\n\n"
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
    
    def _get_node_parser_for_content(self, content: str, file_path: str = None) -> Union[MarkdownTokenTextSplitter, UnicodeSafeTokenTextSplitter, SentenceSplitter]:
        """
        Determine the appropriate node parser based on content type and file extension.
        
        Args:
            content: The text content to analyze
            file_path: Optional file path to help determine content type
            
        Returns:
            The appropriate node parser instance
        """
        # # Check if it's a markdown file
        # if file_path and file_path.lower().endswith(('.md', '.markdown', '.mdown')):
        #     return self.markdown_splitter
        #
        # # Check if content contains markdown patterns
        # markdown_patterns = [
        #     r'^#+\s+',  # Headers
        #     r'^\s*[-*+]\s+',  # Unordered lists
        #     r'^\s*\d+\.\s+',  # Ordered lists
        #     r'\|.*\|',  # Tables
        #     r'\[.*\]\(.*\)',  # Links
        #     r'\*\*.*\*\*',  # Bold text
        #     r'\*.*\*',  # Italic text
        #     r'`.*`',  # Inline code
        #     r'```',  # Code blocks
        # ]
        #
        # import re
        # for pattern in markdown_patterns:
        #     if re.search(pattern, content, re.MULTILINE):
        #         return self.markdown_splitter
        #
        # # Default to UnicodeSafeTokenTextSplitter for all other content
        # # This ensures safe handling of any Unicode characters that might be present
        # return self.unicode_splitter

        return self.markdown_splitter
    
    def _create_service_context_with_parser(self, node_parser) -> ServiceContext:
        """
        Create a service context with a specific node parser.
        
        Args:
            node_parser: The node parser to use
            
        Returns:
            ServiceContext with the specified parser
        """
        return ServiceContext.from_defaults(
            embed_model=self.embedding_model,
            node_parser=node_parser,
            llm=None
        )
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a single text document using LlamaIndex with appropriate custom splitter.
        
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
            
            # Determine the appropriate node parser
            node_parser = self._get_node_parser_for_content(text, metadata.get('file_path') if metadata else None)
            logger.info(10 * "*")
            logger.info(node_parser)
            logger.info(10 * "*")
            # Create LlamaIndex Document with the generated ID
            document = Document(
                text=text,
                metadata=metadata or {},
                id_=doc_id
            )
            
            # Insert document into the existing index
            # The index will use the service context that was configured during initialization
            self.index.insert(document)
            
            logger.info(f"Ingested text document with ID: {doc_id} using {node_parser.__class__.__name__}")
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

            return self.ingest_text(content, file_metadata)
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            raise
    
    def ingest_markdown_directory(self, data_dir: str, recursive: bool = True) -> List[str]:
        """
        Recursively ingest all markdown files from a directory using custom markdown splitter.
        
        Args:
            data_dir: Path to the directory containing markdown files
            recursive: Whether to process subdirectories recursively
            
        Returns:
            List[str]: List of document IDs that were inserted
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path is not a directory: {data_dir}")
        
        markdown_extensions = {'.md', '.markdown', '.mdown'}
        doc_ids = []
        processed_files = 0
        
        logger.info(f"Starting markdown ingestion from directory: {data_dir} using custom markdown splitter")
        
        for root, dirs, files in os.walk(data_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if any(file.lower().endswith(ext) for ext in markdown_extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Create metadata with directory structure info
                        relative_path = os.path.relpath(file_path, data_dir)
                        dir_name = os.path.dirname(relative_path)
                        
                        metadata = {
                            'source_directory': data_dir,
                            'relative_path': relative_path,
                            'directory': dir_name,
                            'file_type': 'markdown',
                            'splitter_used': 'MarkdownTokenTextSplitter'  # Track which splitter was used
                        }
                        
                        doc_id = self.ingest_file(file_path, metadata)
                        doc_ids.append(doc_id)
                        processed_files += 1
                        
                        logger.info(f"Processed markdown file: {relative_path}")
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
            
            # Stop recursion if not requested
            if not recursive:
                break
        
        logger.info(f"Completed markdown ingestion. Processed {processed_files} files.")
        return doc_ids
    
    def ingest_documents(self, documents: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """
        Ingest multiple documents using LlamaIndex with appropriate custom splitters.
        
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
                        metadata->>'splitter_used' as splitter,
                        COUNT(*) as count
                    FROM {Config.DOCUMENTS_TABLE}
                    WHERE metadata->>'splitter_used' IS NOT NULL
                    GROUP BY metadata->>'splitter_used'
                    ORDER BY count DESC
                """)
                results = cursor.fetchall()
            
            conn.close()
            
            stats = {row[0]: row[1] for row in results}
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
