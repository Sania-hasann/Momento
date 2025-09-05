"""
Document retrieval module for the RAG pipeline using LlamaIndex.
Enhanced with BM25 keyword search for maximum hybrid search effectiveness.
"""
import logging
import math
import re
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
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
    
    def __init__(self, bm25_k1: float = None, bm25_b: float = None):
        """Initialize the retrieval pipeline with LlamaIndex and BM25."""
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize node parser
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
        
        # Initialize retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=Config.SIMILARITY_TOP_K
        )
        
        # Initialize query engine with simpler configuration
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever
        )
        
        # Initialize BM25 search
        self.bm25_k1 = bm25_k1 if bm25_k1 is not None else Config.BM25_K1
        self.bm25_b = bm25_b if bm25_b is not None else Config.BM25_B
        self.bm25_doc_freqs = defaultdict(int)
        self.bm25_idf = {}
        self.bm25_doc_lengths = {}
        self.bm25_doc_terms = {}
        self.bm25_avg_doc_length = 0.0
        self.bm25_total_docs = 0
        self._build_bm25_index()
        
        logger.info(f"Initialized Retrieval with LlamaIndex, BM25, and model: {Config.EMBEDDING_MODEL}")
    
    def search(self, query: str, k: int = None, hybrid_weight: float = None) -> List[Dict[str, Any]]:
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
            k = k if k is not None else Config.DEFAULT_K
            hybrid_weight = hybrid_weight if hybrid_weight is not None else Config.DEFAULT_HYBRID_WEIGHT
            return self.true_hybrid_search(query, k, hybrid_weight)
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    def vector_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using direct PostgreSQL queries with pgvector.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with similarity scores
        """
        try:
            k = k if k is not None else Config.DEFAULT_K
            # Create query embedding
            query_embedding = self.embedding_model.get_text_embedding(query)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Use direct PostgreSQL query with pgvector
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Use cosine similarity with pgvector
                cursor.execute(f"""
                    SELECT 
                        node_id as id,
                        text as content,
                        metadata_,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM {Config.DOCUMENTS_TABLE}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, embedding_str, k * Config.VECTOR_SEARCH_MULTIPLIER))
                
                results = cursor.fetchall()
            
            conn.close()
            
            # Process results
            documents = []
            seen_contents = set()
            
            for row in results:
                # Parse metadata from the complex metadata_ structure
                metadata = {}
                if row['metadata_']:
                    if 'topic' in row['metadata_']:
                        metadata['topic'] = row['metadata_']['topic']
                    if 'difficulty' in row['metadata_']:
                        metadata['difficulty'] = row['metadata_']['difficulty']
                    if 'category' in row['metadata_']:
                        metadata['category'] = row['metadata_']['category']
                    if 'length' in row['metadata_']:
                        metadata['length'] = row['metadata_']['length']
                    if 'file_name' in row['metadata_']:
                        metadata['file_name'] = row['metadata_']['file_name']
                    if 'directory' in row['metadata_']:
                        metadata['directory'] = row['metadata_']['directory']
                
                # Create content hash for deduplication
                content_preview = row['content'][:Config.CONTENT_PREVIEW_LENGTH]
                content_hash = hash(content_preview)
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    
                    documents.append({
                        'id': row['id'],
                        'content': row['content'],
                        'metadata': metadata,
                        'similarity': float(row['similarity']),
                        'distance': 1.0 - float(row['similarity']),
                        'combined_score': float(row['similarity'])  # For compatibility
                    })
                    
                    if len(documents) >= k:
                        break
            
            logger.info(f"Retrieved {len(documents)} documents using direct PostgreSQL vector search")
            return documents
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            # Fallback to BM25 search if vector search fails
            logger.info("Falling back to BM25 search due to vector search error")
            return self.bm25_search(query, k)
    
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
            # Try to get document from docstore first
            try:
                doc = self.index.docstore.get_document(doc_id)
                if doc:
                    return {
                        'id': doc_id,
                        'content': doc.text,
                        'metadata': doc.metadata,
                        'created_at': None  # LlamaIndex doesn't store creation time
                    }
            except Exception:
                pass
            
            # If not found in docstore, try to find it in vector store results
            # This is a workaround for when documents are only in the vector store
            try:
                # Search for the document by doing a broad search and filtering
                results = self.vector_search("", k=100)  # Get many results
                for result in results:
                    if result['id'] == doc_id:
                        return {
                            'id': doc_id,
                            'content': result['content'],
                            'metadata': result['metadata'],
                            'created_at': None
                        }
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            raise
    
    def get_document_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by node ID from the vector store.
        
        Args:
            node_id: The node ID from search results
            
        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
        """
        try:
            # Try to get document directly from LlamaIndex docstore first
            try:
                doc = self.index.docstore.get_document(node_id)
                if doc:
                    return {
                        'id': node_id,
                        'content': doc.text,
                        'metadata': doc.metadata,
                        'created_at': None
                    }
            except Exception:
                pass
            
            # Fallback: Search for document by content similarity
            # Get a sample of documents to search through
            search_results = self.vector_search("", k=Config.MAX_SEARCH_CANDIDATES)
            
            for result in search_results:
                if result['id'] == node_id:
                    # Found the document, now get its full details from database
                    return self._get_document_by_content(result['content'])
            
            # If still not found, try direct database query
            return self._get_document_by_node_id_direct(node_id)
            
        except Exception as e:
            logger.error(f"Error retrieving document by node ID {node_id}: {e}")
            return None
    
    def _get_document_by_node_id_direct(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document directly from database using node_id.
        
        Args:
            node_id: The node ID to search for
            
        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
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
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Try to find document by node_id in the data_documents table
                cursor.execute("""
                    SELECT id, text as content, metadata_, node_id
                    FROM data_documents 
                    WHERE node_id = %s
                    LIMIT 1
                """, (node_id,))
                
                result = cursor.fetchone()
                if result:
                    # Parse the metadata from the metadata_ field
                    metadata = {}
                    if result['metadata_']:
                        if 'topic' in result['metadata_']:
                            metadata['topic'] = result['metadata_']['topic']
                        if 'difficulty' in result['metadata_']:
                            metadata['difficulty'] = result['metadata_']['difficulty']
                        if 'category' in result['metadata_']:
                            metadata['category'] = result['metadata_']['category']
                        if 'length' in result['metadata_']:
                            metadata['length'] = result['metadata_']['length']
                    
                    return {
                        'id': result['node_id'],
                        'content': result['content'],
                        'metadata': metadata,
                        'created_at': None
                    }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by node ID direct: {e}")
            return None
    
    def _get_document_by_content(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its content from the database.
        
        Args:
            content: The document content to search for
            
        Returns:
            Optional[Dict[str, Any]]: The document if found, None otherwise
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
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Search for document by content in the data_documents table
                content_start = content[:100]  # Use first 100 characters for matching
                cursor.execute("""
                    SELECT id, text as content, metadata_, node_id
                    FROM data_documents 
                    WHERE text LIKE %s
                    LIMIT 1
                """, (f"{content_start}%",))
                
                result = cursor.fetchone()
                if result:
                    # Parse the metadata from the metadata_ field
                    metadata = {}
                    if result['metadata_']:
                        # Extract the actual metadata from the complex metadata_ structure
                        if 'topic' in result['metadata_']:
                            metadata['topic'] = result['metadata_']['topic']
                        if 'difficulty' in result['metadata_']:
                            metadata['difficulty'] = result['metadata_']['difficulty']
                    
                    return {
                        'id': result['node_id'],  # Use the node_id as the document ID
                        'content': result['content'],
                        'metadata': metadata,
                        'created_at': None  # LlamaIndex doesn't store creation time in this table
                    }
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by content: {e}")
            return None
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the index.
        
        Returns:
            List[Dict[str, Any]]: List of all documents with their IDs
        """
        try:
            documents = []
            # Get all document IDs from the docstore
            all_doc_ids = list(self.index.docstore.docs.keys())
            
            for doc_id in all_doc_ids:
                try:
                    doc = self.index.docstore.get_document(doc_id)
                    if doc:
                        documents.append({
                            'id': doc_id,
                            'content': doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
                            'metadata': doc.metadata
                        })
                except Exception as e:
                    logger.warning(f"Could not retrieve document {doc_id}: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def close(self):
        """Close the vector store connection."""
        try:
            # The vector store will handle its own connection cleanup
            logger.info("Retrieval connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def _remove_duplicates(self, results: List[Dict], similarity_threshold: float = None) -> List[Dict]:
        """
        Remove duplicate or highly similar results based on content and metadata.
        
        Args:
            results: List of search results
            similarity_threshold: Threshold for considering results as duplicates
            
        Returns:
            List[Dict]: Results with duplicates removed
        """
        if not results:
            return results
        
        similarity_threshold = similarity_threshold if similarity_threshold is not None else Config.SIMILARITY_THRESHOLD
        unique_results = []
        seen_content_hashes = set()
        
        for result in results:
            # Create content hash for duplicate detection (use more content for better uniqueness)
            content = result.get('content', '').lower().strip()
            content_hash = hash(content[:Config.CONTENT_HASH_LENGTH])  # Use first N chars for hash
            
            # Check if this is a duplicate (only check exact content matches)
            is_duplicate = False
            
            # Only check for exact content matches
            if content_hash in seen_content_hashes:
                is_duplicate = True
            
            # Add if not duplicate
            if not is_duplicate:
                unique_results.append(result)
                seen_content_hashes.add(content_hash)
        
        return unique_results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def true_hybrid_search(self, query: str, k: int = None, vector_weight: float = None) -> List[Dict[str, Any]]:
        """
        Perform true hybrid search combining vector similarity and full-text search with improved results and duplicate removal.
        
        Args:
            query: The search query
            k: Number of top results to return
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: List of documents with combined scores and duplicates removed
        """
        try:
            k = k if k is not None else Config.DEFAULT_K
            vector_weight = vector_weight if vector_weight is not None else Config.DEFAULT_VECTOR_WEIGHT
            
            # 1. Vector Search - get more candidates for better diversity
            vector_results = self.vector_search(query, k * Config.HYBRID_SEARCH_MULTIPLIER)
            
            # 2. Full-text Search using PostgreSQL - get more candidates
            text_results = self._postgres_full_text_search(query, k * Config.HYBRID_SEARCH_MULTIPLIER)
            
            # 3. Combine and re-rank results with improved algorithm
            combined_results = self._combine_search_results(
                vector_results, text_results, vector_weight
            )
            
            # 4. Remove duplicates for better diversity (less aggressive threshold)
            deduplicated_results = self._remove_duplicates(combined_results, similarity_threshold=Config.HYBRID_SIMILARITY_THRESHOLD)
            
            # 5. Return top-k results
            final_results = deduplicated_results[:k]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    def _postgres_full_text_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform full-text search using PostgreSQL's built-in full-text search with improved keyword matching.
        
        Args:
            query: The search query
            k: Number of top results to return
            
        Returns:
            List[Dict[str, Any]]: List of documents with text relevance scores
        """
        try:
            k = k if k is not None else Config.DEFAULT_K
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                host=Config.DB_HOST,
                port=Config.DB_PORT,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD
            )
            
            # First, try to create full-text search index if it doesn't exist
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_data_documents_text_fts 
                        ON data_documents 
                        USING gin(to_tsvector('english', text));
                    """)
                conn.commit()
            except Exception as e:
                logger.warning(f"Could not create full-text search index: {e}")
            
            # Split query into individual keywords for better matching
            keywords = query.lower().split()
            
            # Create multiple search strategies for better coverage
            search_conditions = []
            search_params = []
            
            # Strategy 1: Full phrase search
            search_conditions.append("to_tsvector('english', text) @@ plainto_tsquery('english', %s)")
            search_params.append(query)
            
            # Strategy 2: Individual keyword search
            for keyword in keywords:
                if len(keyword) > Config.MIN_KEYWORD_LENGTH:  # Only search for keywords with N+ characters
                    search_conditions.append("text ILIKE %s")
                    search_params.append(f"%{keyword}%")
            
            # Strategy 3: Web search for better keyword matching
            search_conditions.append("to_tsvector('english', text) @@ websearch_to_tsquery('english', %s)")
            search_params.append(query)
            
            # Combine all search conditions
            where_clause = " OR ".join(search_conditions)
            
            # Create improved full-text search query
            sql_query = f"""
                SELECT 
                    node_id as id,
                    text as content,
                    metadata_,
                    ts_rank(to_tsvector('english', text), plainto_tsquery('english', %s)) as text_score,
                    CASE 
                        WHEN text ILIKE %s THEN 1.0
                        ELSE 0.0
                    END as exact_match_bonus
                FROM data_documents 
                WHERE {where_clause}
                ORDER BY text_score DESC, exact_match_bonus DESC
                LIMIT %s
            """
            
            # Add parameters for the query
            all_params = [query, f"%{query}%"] + search_params + [k * Config.FULL_TEXT_SEARCH_MULTIPLIER]  # Get more results for deduplication
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql_query, all_params)
                results = cursor.fetchall()
            
            conn.close()
            
            # Convert to standard format and apply deduplication
            documents = []
            seen_contents = set()
            
            for row in results:
                # Parse metadata from the complex metadata_ structure
                metadata = {}
                if row['metadata_']:
                    if 'topic' in row['metadata_']:
                        metadata['topic'] = row['metadata_']['topic']
                    if 'difficulty' in row['metadata_']:
                        metadata['difficulty'] = row['metadata_']['difficulty']
                    if 'category' in row['metadata_']:
                        metadata['category'] = row['metadata_']['category']
                    if 'length' in row['metadata_']:
                        metadata['length'] = row['metadata_']['length']
                
                # Create content hash for deduplication
                content_preview = row['content'][:Config.CONTENT_PREVIEW_LENGTH]
                content_hash = hash(content_preview)
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    
                    # Calculate enhanced score with exact match bonus
                    base_score = float(row['text_score']) if row['text_score'] else 0.0
                    exact_bonus = float(row['exact_match_bonus']) if row['exact_match_bonus'] else 0.0
                    enhanced_score = min(1.0, base_score + (exact_bonus * Config.EXACT_MATCH_BONUS))
                    
                    documents.append({
                        'id': row['id'],
                        'content': row['content'],
                        'metadata': metadata,
                        'text_score': enhanced_score,
                        'similarity': enhanced_score,
                        'distance': 1.0 - enhanced_score,
                        'combined_score': enhanced_score
                    })
                    
                    if len(documents) >= k:
                        break
            
            logger.info(f"Retrieved {len(documents)} documents using improved full-text search")
            return documents
            
        except Exception as e:
            logger.error(f"Error performing full-text search: {e}")
            return []

    def _normalize_scores(self, results: List[Dict], score_field: str) -> List[Dict]:
        """
        Normalize scores to 0-1 range for better differentiation with improved scaling.
        
        Args:
            results: List of search results
            score_field: Field name containing the score to normalize
            
        Returns:
            List[Dict]: Results with normalized scores
        """
        if not results:
            return results
        
        # Extract scores
        scores = []
        for result in results:
            if score_field in result and result[score_field] is not None:
                scores.append(result[score_field])
        
        if not scores:
            return results
        
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score
        
        # Normalize scores with improved scaling
        for result in results:
            if score_field in result and result[score_field] is not None:
                if score_range == 0:
                    # All scores are the same, assign equal weights
                    result[f'normalized_{score_field}'] = Config.DEFAULT_NORMALIZED_SCORE
                else:
                    # Normalize to 0-1 range with sigmoid-like scaling for better differentiation
                    normalized = (result[score_field] - min_score) / score_range
                    # Apply sigmoid-like transformation to spread out scores
                    result[f'normalized_{score_field}'] = 1 / (1 + math.exp(-Config.SIGMOID_SCALING_FACTOR * (normalized - 0.5)))
        
        return results

    def _deduplicate_results(self, results: List[Dict], k: int) -> List[Dict]:
        """
        Remove duplicate results based on content similarity.
        
        Args:
            results: List of search results
            k: Number of results to return
            
        Returns:
            List[Dict]: Deduplicated results
        """
        seen_contents = set()
        unique_results = []
        
        for result in results:
            # Create content hash for deduplication
            content_preview = result['content'][:Config.CONTENT_PREVIEW_LENGTH]
            content_hash = hash(content_preview)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
                
                if len(unique_results) >= k:
                    break
        
        return unique_results

    def _combine_search_results(self, vector_results: List[Dict], text_results: List[Dict], 
                               vector_weight: float = None) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results with improved weighted scoring and deduplication.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from full-text search
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: Combined and re-ranked results
        """
        vector_weight = vector_weight if vector_weight is not None else Config.DEFAULT_VECTOR_WEIGHT
        
        # Normalize scores for better combination
        vector_results = self._normalize_scores(vector_results, 'similarity')
        text_results = self._normalize_scores(text_results, 'text_score')
        
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
                    'vector_score': vector_doc.get('normalized_similarity', vector_doc['similarity']),
                    'vector_distance': vector_doc['distance']
                })
            
            # Add text search data
            if text_doc:
                combined_doc.update({
                    'content': text_doc['content'],
                    'metadata': text_doc['metadata'],
                    'text_score': text_doc.get('normalized_text_score', text_doc['text_score']),
                    'text_distance': text_doc['distance']
                })
            
            # Calculate combined score with improved weighting
            text_weight = 1.0 - vector_weight
            
            # Use normalized scores if available, otherwise use original scores
            vector_score = combined_doc['vector_score']
            text_score = combined_doc['text_score']
            
            combined_doc['combined_score'] = (
                vector_weight * vector_score +
                text_weight * text_score
            )
            
            # Add compatibility fields
            combined_doc['similarity'] = combined_doc['combined_score']
            combined_doc['distance'] = 1.0 - combined_doc['combined_score']
            combined_doc['score'] = combined_doc['combined_score']
            
            combined_results.append(combined_doc)
        
        # Sort by combined score (descending)
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Apply deduplication
        combined_results = self._deduplicate_results(combined_results, len(combined_results))
        
        logger.info(f"Combined {len(combined_results)} documents with vector_weight={vector_weight}")
        return combined_results
    
    def _build_bm25_index(self):
        """Build BM25 index from the database."""
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
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get all documents
                cursor.execute(f"""
                    SELECT node_id, text, metadata_
                    FROM {Config.DOCUMENTS_TABLE}
                """)
                
                documents = cursor.fetchall()
                self.bm25_total_docs = len(documents)
                
                if self.bm25_total_docs == 0:
                    logger.warning("No documents found in database for BM25 indexing")
                    return
                
                # Process each document
                total_length = 0
                for doc in documents:
                    doc_id = doc['node_id']
                    text = doc['text']
                    
                    # Tokenize and preprocess text
                    terms = self._bm25_tokenize(text)
                    self.bm25_doc_terms[doc_id] = terms
                    self.bm25_doc_lengths[doc_id] = len(terms)
                    total_length += len(terms)
                    
                    # Count term frequencies in this document
                    term_counts = Counter(terms)
                    for term in term_counts:
                        self.bm25_doc_freqs[term] += 1
                
                # Calculate average document length
                self.bm25_avg_doc_length = total_length / self.bm25_total_docs if self.bm25_total_docs > 0 else 0
                
                # Calculate IDF for each term
                for term, doc_freq in self.bm25_doc_freqs.items():
                    # IDF = log((N - df + 0.5) / (df + 0.5))
                    # where N = total documents, df = document frequency
                    self.bm25_idf[term] = math.log(
                        (self.bm25_total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
                    )
                
                logger.info(f"BM25 index built: {self.bm25_total_docs} documents, "
                          f"{len(self.bm25_doc_freqs)} unique terms, "
                          f"avg doc length: {self.bm25_avg_doc_length:.1f}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            # Don't raise - allow pipeline to continue without BM25
    
    def _bm25_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for BM25 processing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of normalized terms
        """
        if not text:
            return []
        
        # Convert to lowercase and split on non-alphanumeric characters
        terms = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        
        # Filter out very short terms (less than 2 characters)
        terms = [term for term in terms if len(term) >= 2]
        
        return terms
    
    def bm25_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword-based search.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of documents with BM25 scores and metadata
        """
        try:
            if self.bm25_total_docs == 0:
                logger.warning("No documents indexed for BM25 search")
                return []
            
            # Tokenize query
            query_terms = self._bm25_tokenize(query)
            if not query_terms:
                logger.warning("No valid terms found in query")
                return []
            
            # Calculate BM25 scores for each document
            doc_scores = {}
            
            for doc_id, doc_terms in self.bm25_doc_terms.items():
                score = 0.0
                doc_length = self.bm25_doc_lengths[doc_id]
                
                # Count term frequencies in this document
                term_counts = Counter(doc_terms)
                
                for term in query_terms:
                    if term in term_counts and term in self.bm25_idf:
                        tf = term_counts[term]  # Term frequency in document
                        idf = self.bm25_idf[term]    # Inverse document frequency
                        
                        # BM25 formula
                        # score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                        numerator = tf * (self.bm25_k1 + 1)
                        denominator = tf + self.bm25_k1 * (
                            1 - self.bm25_b + self.bm25_b * (doc_length / self.bm25_avg_doc_length)
                        )
                        score += idf * (numerator / denominator)
                
                if score > 0:
                    doc_scores[doc_id] = score
            
            # Sort by score and get top-k results
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get document details for top results
            results = []
            for doc_id, score in sorted_docs[:k]:
                doc_info = self._get_document_by_node_id_direct(doc_id)
                if doc_info:
                    doc_info['bm25_score'] = score
                    doc_info['score'] = score  # For compatibility
                    results.append(doc_info)
            
            logger.info(f"BM25 search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            return []
    
    def enhanced_hybrid_search(
        self, 
        query: str, 
        k: int = 5, 
        vector_weight: float = None,
        fusion_method: str = None,
        query_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform enhanced hybrid search combining BM25 + pgvector with advanced fusion.
        
        Args:
            query: Search query string
            k: Number of results to return
            vector_weight: Weight for vector search (0.0-1.0), None for auto
            fusion_method: Fusion method ("wlc", "rrf", "adaptive")
            query_aware: Whether to use query-aware weighting
            
        Returns:
            List of documents with combined scores and metadata
        """
        try:
            # Set defaults
            fusion_method = fusion_method if fusion_method is not None else Config.DEFAULT_FUSION_METHOD
            
            # Determine optimal weights
            if vector_weight is None:
                vector_weight = self._get_optimal_weight(query, query_aware)
            
            # Get results from both methods
            bm25_results = self.bm25_search(query, k * Config.HYBRID_SEARCH_MULTIPLIER)  # Get more candidates
            vector_results = self.vector_search(query, k * Config.HYBRID_SEARCH_MULTIPLIER)
            
            # Apply fusion method
            if fusion_method == "rrf":
                combined_results = self._reciprocal_rank_fusion(
                    [bm25_results, vector_results], k
                )
            elif fusion_method == "adaptive":
                combined_results = self._adaptive_fusion(
                    bm25_results, vector_results, query, k
                )
            else:  # Default: Weighted Linear Combination
                combined_results = self._enhanced_weighted_combination(
                    bm25_results, vector_results, vector_weight, k
                )
            
            # Add fusion metadata
            for result in combined_results:
                result['fusion_method'] = fusion_method
                result['vector_weight'] = vector_weight
                result['query_aware'] = query_aware
            
            logger.info(f"Enhanced hybrid search returned {len(combined_results)} results "
                       f"using {fusion_method} fusion with vector_weight={vector_weight:.2f}")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in enhanced hybrid search: {e}")
            return []
    
    def _get_optimal_weight(self, query: str, query_aware: bool = True) -> float:
        """
        Determine optimal vector weight based on query characteristics.
        
        Args:
            query: Search query
            query_aware: Whether to use query analysis
            
        Returns:
            Optimal vector weight (0.0-1.0)
        """
        if not query_aware:
            return Config.BALANCED_WEIGHT  # Default 70% vector, 30% BM25
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(query)
        
        # Adjust weight based on query type
        if query_analysis['is_keyword_heavy']:
            # More keyword-focused queries benefit from higher BM25 weight
            return Config.KEYWORD_HEAVY_WEIGHT  # 40% vector, 60% BM25
        elif query_analysis['is_semantic_heavy']:
            # More semantic queries benefit from higher vector weight
            return Config.SEMANTIC_HEAVY_WEIGHT  # 80% vector, 20% BM25
        else:
            # Balanced queries use default weight
            return Config.BALANCED_WEIGHT
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics to determine optimal search strategy.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Dictionary with query analysis results
        """
        # Basic analysis
        words = query.lower().split()
        word_count = len(words)
        
        # Check for keyword indicators
        has_quotes = '"' in query
        has_specific_terms = any(word in query.lower() for word in [
            'exact', 'precise', 'specific', 'exactly', 'precisely'
        ])
        has_technical_terms = any(word in query.lower() for word in [
            'api', 'sql', 'json', 'http', 'rest', 'graphql', 'docker', 'kubernetes'
        ])
        
        # Check for semantic indicators
        has_conceptual_terms = any(word in query.lower() for word in [
            'concept', 'idea', 'meaning', 'similar', 'related', 'like', 'about'
        ])
        has_question_words = any(word in query.lower() for word in [
            'what', 'how', 'why', 'when', 'where', 'which', 'who'
        ])
        
        # Determine query type
        keyword_score = 0
        semantic_score = 0
        
        if has_quotes or has_specific_terms:
            keyword_score += 2
        if has_technical_terms:
            keyword_score += 1
        if word_count <= 2:
            keyword_score += 1
        
        if has_conceptual_terms or has_question_words:
            semantic_score += 2
        if word_count > 4:
            semantic_score += 1
        
        return {
            'word_count': word_count,
            'is_keyword_heavy': keyword_score > semantic_score,
            'is_semantic_heavy': semantic_score > keyword_score,
            'keyword_score': keyword_score,
            'semantic_score': semantic_score,
            'has_quotes': has_quotes,
            'has_technical_terms': has_technical_terms
        }
    
    def _enhanced_weighted_combination(
        self, 
        bm25_results: List[Dict], 
        vector_results: List[Dict], 
        vector_weight: float, 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using enhanced weighted linear combination.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            vector_weight: Weight for vector search (0.0-1.0)
            k: Number of results to return
            
        Returns:
            Combined and ranked results
        """
        # Normalize scores
        bm25_results = self._normalize_scores(bm25_results, 'bm25_score')
        vector_results = self._normalize_scores(vector_results, 'similarity')
        
        # Create lookup dictionaries
        bm25_lookup = {doc['id']: doc for doc in bm25_results}
        vector_lookup = {doc['id']: doc for doc in vector_results}
        
        # Get all unique document IDs
        all_ids = set(bm25_lookup.keys()) | set(vector_lookup.keys())
        
        combined_results = []
        for doc_id in all_ids:
            bm25_doc = bm25_lookup.get(doc_id)
            vector_doc = vector_lookup.get(doc_id)
            
            # Initialize combined document
            combined_doc = {
                'id': doc_id,
                'content': '',
                'metadata': {},
                'bm25_score': 0.0,
                'vector_score': 0.0,
                'combined_score': 0.0
            }
            
            # Add BM25 data
            if bm25_doc:
                combined_doc.update({
                    'content': bm25_doc['content'],
                    'metadata': bm25_doc['metadata'],
                    'bm25_score': bm25_doc.get('normalized_bm25_score', bm25_doc['bm25_score'])
                })
            
            # Add vector data
            if vector_doc:
                combined_doc.update({
                    'content': vector_doc['content'],
                    'metadata': vector_doc['metadata'],
                    'vector_score': vector_doc.get('normalized_similarity', vector_doc['similarity'])
                })
            
            # Calculate combined score
            bm25_weight = 1.0 - vector_weight
            combined_doc['combined_score'] = (
                bm25_weight * combined_doc['bm25_score'] +
                vector_weight * combined_doc['vector_score']
            )
            
            # Add compatibility fields
            combined_doc['score'] = combined_doc['combined_score']
            combined_doc['similarity'] = combined_doc['combined_score']
            
            combined_results.append(combined_doc)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:k]
    
    def _reciprocal_rank_fusion(
        self, 
        results_lists: List[List[Dict]], 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF is more robust to different score distributions and often
        performs better than simple weighted combinations.
        
        Args:
            results_lists: List of result lists from different search methods
            k: Number of results to return
            
        Returns:
            Combined results ranked by RRF scores
        """
        doc_scores = {}
        rrf_k = Config.RRF_K  # RRF parameter (typically 60)
        
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = doc['id']
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (rrf_k + rank + 1)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get document details for top results
        combined_results = []
        for doc_id, rrf_score in sorted_docs[:k]:
            # Find the document in any of the result lists
            doc_info = None
            for results in results_lists:
                for doc in results:
                    if doc['id'] == doc_id:
                        doc_info = doc.copy()
                        break
                if doc_info:
                    break
            
            if doc_info:
                doc_info['rrf_score'] = rrf_score
                doc_info['combined_score'] = rrf_score
                doc_info['score'] = rrf_score
                doc_info['similarity'] = rrf_score
                combined_results.append(doc_info)
        
        return combined_results
    
    def _adaptive_fusion(
        self, 
        bm25_results: List[Dict], 
        vector_results: List[Dict], 
        query: str, 
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Adaptive fusion that dynamically adjusts weights based on result quality.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            query: Original search query
            k: Number of results to return
            
        Returns:
            Adaptively fused results
        """
        # Analyze result quality for each method
        bm25_quality = self._assess_result_quality(bm25_results, query)
        vector_quality = self._assess_result_quality(vector_results, query)
        
        # Adjust weights based on quality
        total_quality = bm25_quality + vector_quality
        if total_quality > 0:
            adaptive_vector_weight = vector_quality / total_quality
        else:
            adaptive_vector_weight = Config.BALANCED_WEIGHT  # Default weight
        
        # Use adaptive weight for combination
        return self._enhanced_weighted_combination(
            bm25_results, vector_results, adaptive_vector_weight, k
        )
    
    def _assess_result_quality(self, results: List[Dict], query: str) -> float:
        """
        Assess the quality of search results for a given query.
        
        Args:
            results: Search results to assess
            query: Original search query
            
        Returns:
            Quality score (0.0-1.0)
        """
        if not results:
            return 0.0
        
        # Simple quality metrics
        score_variance = 0.0
        if len(results) > 1:
            scores = [r.get('score', 0.0) for r in results]
            score_variance = max(scores) - min(scores)
        
        # Higher variance indicates better differentiation
        quality_score = min(1.0, score_variance * 2)
        
        return quality_score
    
    def get_bm25_stats(self) -> Dict[str, Any]:
        """
        Get BM25 index statistics.
        
        Returns:
            Dictionary with BM25 statistics
        """
        return {
            'total_documents': self.bm25_total_docs,
            'unique_terms': len(self.bm25_doc_freqs),
            'average_doc_length': self.bm25_avg_doc_length,
            'k1_parameter': self.bm25_k1,
            'b_parameter': self.bm25_b
        }
    
    def rebuild_bm25_index(self):
        """Rebuild the BM25 index from scratch."""
        logger.info("Rebuilding BM25 index...")
        self.bm25_doc_freqs.clear()
        self.bm25_idf.clear()
        self.bm25_doc_lengths.clear()
        self.bm25_doc_terms.clear()
        self.bm25_avg_doc_length = 0.0
        self.bm25_total_docs = 0
        self._build_bm25_index()
        logger.info("BM25 index rebuilt successfully")
