"""
Document retrieval module for the RAG pipeline using LlamaIndex.
"""
import logging
from typing import List, Dict, Any, Optional
from llama_index.legacy import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.node_parser import SentenceSplitter
from llama_index.legacy.retrievers import VectorIndexRetriever
from llama_index.legacy.query_engine import RetrieverQueryEngine
from llama_index.legacy.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from config import Config

logger = logging.getLogger(__name__)

class Retrieval:
    """
    Handles document retrieval using LlamaIndex with hybrid search capabilities.
    
    This class implements retrieval using LlamaIndex's query engine and retriever
    components, providing both vector similarity search and hybrid search options.
    """
    
    def __init__(self):
        """Initialize the retrieval pipeline with LlamaIndex."""
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
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
        
        # Initialize retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10
        )
        
        # Initialize query engine with simpler configuration
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever
        )
        
        logger.info(f"Initialized Retrieval with LlamaIndex and model: {Config.EMBEDDING_MODEL}")
    
    def search(self, query: str, k: int = 5, hybrid_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and full-text search.
        
        Args:
            query: The search query
            k: Number of top results to return
            hybrid_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: List of documents with combined scores and metadata
        """
        try:
            # Use true hybrid search with the provided weight
            return self.true_hybrid_search(query, k, hybrid_weight)
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using LlamaIndex's vector store directly.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with similarity scores
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.get_text_embedding(query)
            
            # Create VectorStoreQuery
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=k,
                mode=VectorStoreQueryMode.DEFAULT
            )
            
            # Query the vector store directly
            results = self.vector_store.query(vector_store_query)
            
            # Extract results from VectorStoreQueryResult
            documents = []
            if results.nodes:
                for i, node in enumerate(results.nodes):
                    score = results.similarities[i] if results.similarities and i < len(results.similarities) else 1.0
                    documents.append({
                        'id': node.node_id,
                        'content': node.text,
                        'metadata': node.metadata,
                        'similarity': score,
                        'distance': 1.0 - score,
                        'combined_score': score  # For compatibility with example script
                    })
            
            logger.info(f"Retrieved {len(documents)} documents using vector search")
            return documents
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            raise
    
    def full_text_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform full-text search using PostgreSQL's built-in full-text search.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with relevance scores
        """
        try:
            return self._postgres_full_text_search(query, k)
            
        except Exception as e:
            logger.error(f"Error performing full-text search: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID using LlamaIndex.
        
        Args:
            doc_id: The document ID
            
        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
        """
        try:
            # Get document from index
            doc = self.index.docstore.get_document(doc_id)
            if doc:
                return {
                    'id': doc_id,
                    'content': doc.text,
                    'metadata': doc.metadata,
                    'created_at': None  # LlamaIndex doesn't store creation time
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            raise
    
    def close(self):
        """Close the vector store connection."""
        try:
            # The vector store will handle its own connection cleanup
            logger.info("Retrieval connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def true_hybrid_search(self, query: str, k: int = 5, vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform true hybrid search combining vector similarity and full-text search.
        
        Args:
            query: The search query
            k: Number of top results to return
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: List of documents with combined scores
        """
        try:
            # 1. Vector Search
            vector_results = self.vector_search(query, k * 2)
            
            # 2. Full-text Search using PostgreSQL
            text_results = self._postgres_full_text_search(query, k * 2)
            
            # 3. Combine and re-rank results
            combined_results = self._combine_search_results(
                vector_results, text_results, vector_weight
            )
            
            # 4. Return top-k results
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    def _postgres_full_text_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform full-text search using PostgreSQL's built-in full-text search.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with text relevance scores
        """
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            
            # Create full-text search query
            # Using the actual column names from your documents table
            sql_query = """
                SELECT 
                    id,
                    content,
                    metadata,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as text_score
                FROM documents 
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY text_score DESC
                LIMIT %s
            """
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql_query, (query, query, k))
                results = cursor.fetchall()
            
            conn.close()
            
            # Convert to standard format
            documents = []
            for row in results:
                documents.append({
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': row['metadata'],
                    'text_score': float(row['text_score']),
                    'similarity': float(row['text_score']),
                    'distance': 1.0 - float(row['text_score']),
                    'combined_score': float(row['text_score'])
                })
            
            logger.info(f"Retrieved {len(documents)} documents using full-text search")
            return documents
            
        except Exception as e:
            logger.error(f"Error performing full-text search: {e}")
            return []
    
    def _combine_search_results(self, vector_results: List[Dict], text_results: List[Dict], 
                               vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results with weighted scoring.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from full-text search
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: Combined and re-ranked results
        """
        # Create lookup dictionaries for quick access
        vector_lookup = {doc['id']: doc for doc in vector_results}
        text_lookup = {doc['id']: doc for doc in text_results}
        
        # Get all unique document IDs
        all_ids = set(vector_lookup.keys()) | set(text_lookup.keys())
        
        combined_results = []
        
        for doc_id in all_ids:
            vector_doc = vector_lookup.get(doc_id)
            text_doc = text_lookup.get(doc_id)
            
            # Initialize combined document
            combined_doc = {
                'id': doc_id,
                'content': '',
                'metadata': {},
                'vector_score': 0.0,
                'text_score': 0.0,
                'combined_score': 0.0
            }
            
            # Add vector search data
            if vector_doc:
                combined_doc.update({
                    'content': vector_doc['content'],
                    'metadata': vector_doc['metadata'],
                    'vector_score': vector_doc['similarity'],
                    'vector_distance': vector_doc['distance']
                })
            
            # Add text search data
            if text_doc:
                combined_doc.update({
                    'content': text_doc['content'],
                    'metadata': text_doc['metadata'],
                    'text_score': text_doc['text_score'],
                    'text_distance': text_doc['distance']
                })
            
            # Calculate combined score
            text_weight = 1.0 - vector_weight
            combined_doc['combined_score'] = (
                vector_weight * combined_doc['vector_score'] +
                text_weight * combined_doc['text_score']
            )
            
            # Add compatibility fields
            combined_doc['similarity'] = combined_doc['combined_score']
            combined_doc['distance'] = 1.0 - combined_doc['combined_score']
            combined_doc['score'] = combined_doc['combined_score']
            
            combined_results.append(combined_doc)
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        logger.info(f"Combined {len(combined_results)} documents with vector_weight={vector_weight}")
        return combined_results
