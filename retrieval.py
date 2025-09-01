"""
Document retrieval module for the RAG pipeline using LlamaIndex.
"""
import logging
import math
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
        Perform vector similarity search using LlamaIndex's vector store directly with improved deduplication.
        
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
                similarity_top_k=k * 2,  # Get more results for deduplication
                mode=VectorStoreQueryMode.DEFAULT
            )
            
            # Query the vector store directly
            results = self.vector_store.query(vector_store_query)
            
            # Extract results from VectorStoreQueryResult
            documents = []
            seen_contents = set()
            
            if results.nodes:
                for i, node in enumerate(results.nodes):
                    score = results.similarities[i] if results.similarities and i < len(results.similarities) else 1.0
                    
                    # Create content hash for deduplication
                    content_preview = node.text[:100]
                    content_hash = hash(content_preview)
                    
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        
                        documents.append({
                            'id': node.node_id,
                            'content': node.text,
                            'metadata': node.metadata,
                            'similarity': score,
                            'distance': 1.0 - score,
                            'combined_score': score  # For compatibility with example script
                        })
                        
                        if len(documents) >= k:
                            break
            
            logger.info(f"Retrieved {len(documents)} documents using improved vector search")
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
    
    def _remove_duplicates(self, results: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
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
        
        unique_results = []
        seen_content_hashes = set()
        seen_metadata_combinations = set()
        
        for result in results:
            # Create content hash for duplicate detection
            content = result.get('content', '').lower().strip()
            content_hash = hash(content[:200])  # Use first 200 chars for hash
            
            # Create metadata combination
            metadata = result.get('metadata', {})
            metadata_combo = f"{metadata.get('topic', '')}-{metadata.get('category', '')}-{metadata.get('domain', '')}"
            
            # Check if this is a duplicate
            is_duplicate = False
            
            # Check content similarity
            if content_hash in seen_content_hashes:
                is_duplicate = True
            
            # Check metadata similarity
            if metadata_combo in seen_metadata_combinations:
                is_duplicate = True
            
            # Check semantic similarity with existing results
            if not is_duplicate and unique_results:
                for existing_result in unique_results[-3:]:  # Check against last 3 results
                    existing_content = existing_result.get('content', '').lower().strip()
                    
                    # Simple similarity check based on common words
                    content_words = set(content.split()[:20])  # First 20 words
                    existing_words = set(existing_content.split()[:20])
                    
                    if content_words and existing_words:
                        similarity = len(content_words.intersection(existing_words)) / len(content_words.union(existing_words))
                        if similarity > similarity_threshold:
                            is_duplicate = True
                            break
            
            # Add if not duplicate
            if not is_duplicate:
                unique_results.append(result)
                seen_content_hashes.add(content_hash)
                seen_metadata_combinations.add(metadata_combo)
        
        return unique_results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def true_hybrid_search(self, query: str, k: int = 5, vector_weight: float = 0.7) -> List[Dict[str, Any]]:
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
            # 1. Vector Search - get more candidates for better diversity
            vector_results = self.vector_search(query, k * 3)
            
            # 2. Full-text Search using PostgreSQL - get more candidates
            text_results = self._postgres_full_text_search(query, k * 3)
            
            # 3. Combine and re-rank results with improved algorithm
            combined_results = self._combine_search_results(
                vector_results, text_results, vector_weight
            )
            
            # 4. Remove duplicates for better diversity
            deduplicated_results = self._remove_duplicates(combined_results, similarity_threshold=0.7)
            
            # 5. Return top-k results
            final_results = deduplicated_results[:k]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    def _postgres_full_text_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform full-text search using PostgreSQL's built-in full-text search with improved keyword matching.
        
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
                if len(keyword) > 2:  # Only search for keywords with 3+ characters
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
            all_params = [query, f"%{query}%"] + search_params + [k * 2]  # Get more results for deduplication
            
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
                content_preview = row['content'][:100]
                content_hash = hash(content_preview)
                
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    
                    # Calculate enhanced score with exact match bonus
                    base_score = float(row['text_score']) if row['text_score'] else 0.0
                    exact_bonus = float(row['exact_match_bonus']) if row['exact_match_bonus'] else 0.0
                    enhanced_score = min(1.0, base_score + (exact_bonus * 0.3))
                    
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
                    result[f'normalized_{score_field}'] = 0.5
                else:
                    # Normalize to 0-1 range with sigmoid-like scaling for better differentiation
                    normalized = (result[score_field] - min_score) / score_range
                    # Apply sigmoid-like transformation to spread out scores
                    result[f'normalized_{score_field}'] = 1 / (1 + math.exp(-5 * (normalized - 0.5)))
        
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
            content_preview = result['content'][:100]
            content_hash = hash(content_preview)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
                
                if len(unique_results) >= k:
                    break
        
        return unique_results

    def _combine_search_results(self, vector_results: List[Dict], text_results: List[Dict], 
                               vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results with improved weighted scoring and deduplication.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from full-text search
            vector_weight: Weight for vector similarity (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: Combined and re-ranked results
        """
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
