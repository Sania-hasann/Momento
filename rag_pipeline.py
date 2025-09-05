"""
Main RAG pipeline that combines document ingestion and retrieval using LlamaIndex.
"""

import logging
from typing import List, Dict, Any, Union, Optional
from document_ingestion import DocumentIngestion
from retrieval import Retrieval

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that combines document ingestion and retrieval capabilities using LlamaIndex.

    This class provides a unified interface for:
    1. Ingesting documents and generating embeddings using LlamaIndex
    2. Performing vector similarity search using LlamaIndex's query engine
    3. Managing the complete RAG workflow with PostgreSQL + pgvector
    """

    def __init__(self):
        """Initialize the RAG pipeline with LlamaIndex."""
        self.ingestion = DocumentIngestion()
        self.retrieval = Retrieval()
        logger.info("RAG pipeline initialized successfully with LlamaIndex")

    def ingest_documents(
        self, documents: List[Union[str, Dict[str, Any]]]
    ) -> List[str]:
        """
        Ingest multiple documents into the pipeline using LlamaIndex.

        Args:
            documents: List of documents to ingest. Each document can be:
                - A string (text content)
                - A dictionary with 'text' and optional 'metadata' keys
                - A dictionary with 'file_path' and optional 'metadata' keys

        Returns:
            List[str]: List of document IDs that were inserted
        """
        return self.ingestion.ingest_documents(documents)

    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a single text document using LlamaIndex.

        Args:
            text: The text content to ingest
            metadata: Optional metadata dictionary

        Returns:
            str: The ID of the inserted document
        """
        return self.ingestion.ingest_text(text, metadata)

    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Ingest a document from a file path using LlamaIndex.

        Args:
            file_path: Path to the file to ingest
            metadata: Optional metadata dictionary

        Returns:
            str: The ID of the inserted document
        """
        return self.ingestion.ingest_file(file_path, metadata)

    def search(
        self, query: str, k: int = None, hybrid_weight: float = None
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced hybrid search combining BM25 + pgvector (default behavior).

        Args:
            query: The search query
            k: Number of top results to return
            hybrid_weight: Weight for vector search (0.0-1.0), None for auto

        Returns:
            List[Dict[str, Any]]: List of documents with combined scores and metadata
        """
        # Use enhanced hybrid search as the default search method
        k = k if k is not None else Config.DEFAULT_K
        hybrid_weight = hybrid_weight if hybrid_weight is not None else Config.DEFAULT_HYBRID_WEIGHT
        return self.retrieval.enhanced_hybrid_search(
            query, k, vector_weight=hybrid_weight, fusion_method=Config.DEFAULT_FUSION_METHOD, query_aware=True
        )
    
    def enhanced_hybrid_search(
        self, 
        query: str, 
        k: int = None, 
        vector_weight: float = None,
        fusion_method: str = None,
        query_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced hybrid search combining BM25 + pgvector with advanced fusion.

        Args:
            query: The search query
            k: Number of top results to return
            vector_weight: Weight for vector search (0.0-1.0), None for auto
            fusion_method: Fusion method ("wlc", "rrf", "adaptive")
            query_aware: Whether to use query-aware weighting

        Returns:
            List[Dict[str, Any]]: List of documents with combined scores and metadata
        """
        k = k if k is not None else Config.DEFAULT_K
        fusion_method = fusion_method if fusion_method is not None else Config.DEFAULT_FUSION_METHOD
        return self.retrieval.enhanced_hybrid_search(
            query, k, vector_weight, fusion_method, query_aware
        )
    
    def bm25_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword-based search.

        Args:
            query: The search query
            k: Number of top results to return

        Returns:
            List[Dict[str, Any]]: List of documents with BM25 scores
        """
        k = k if k is not None else Config.DEFAULT_K
        return self.retrieval.bm25_search(query, k)

    def vector_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using LlamaIndex's retriever.

        Args:
            query: The search query
            k: Number of top results to return

        Returns:
            List[Dict[str, Any]]: List of documents with similarity scores
        """
        k = k if k is not None else Config.DEFAULT_K
        return self.retrieval.vector_search(query, k)

    def full_text_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform full-text search (falls back to vector search in LlamaIndex).

        Args:
            query: The search query
            k: Number of top results to return

        Returns:
            List[Dict[str, Any]]: List of documents with relevance scores
        """
        k = k if k is not None else Config.DEFAULT_K
        return self.retrieval.full_text_search(query, k)

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID using LlamaIndex.

        Args:
            doc_id: The document ID

        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
        """
        return self.retrieval.get_document_by_id(doc_id)

    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the index.

        Returns:
            List[Dict[str, Any]]: List of all documents with their IDs
        """
        return self.retrieval.list_all_documents()

    def get_document_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by node ID from the vector store.

        Args:
            node_id: The node ID from search results

        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
        """
        return self.retrieval.get_document_by_node_id(node_id)

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.

        Returns:
            int: Number of documents
        """
        return self.ingestion.get_document_count()

    def close(self):
        """Close all connections."""
        self.ingestion.close()
        self.retrieval.close()
        logger.info("RAG pipeline connections closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
