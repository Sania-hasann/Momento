#!/usr/bin/env python3
"""
Example script demonstrating the complete RAG pipeline functionality.

This script shows how to:
1. Set up the database connection
2. Ingest sample documents
3. Perform different types of searches
4. Display results
"""
import logging
import json
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_documents():
    """Create sample documents for demonstration."""
    return [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns.",
            "metadata": {"topic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "text": "Deep learning is a type of machine learning that uses neural networks with multiple layers to model and understand complex patterns. It has been particularly successful in image recognition, natural language processing, and speech recognition.",
            "metadata": {"topic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "text": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.",
            "metadata": {"topic": "nlp", "difficulty": "intermediate"}
        },
        {
            "text": "PostgreSQL is a powerful, open-source object-relational database system. It extends the SQL language with additional features and provides advanced data types, including support for JSON, arrays, and custom types.",
            "metadata": {"topic": "database", "difficulty": "beginner"}
        },
        {
            "text": "Vector databases are specialized database systems designed to store and query high-dimensional vector data efficiently. They are commonly used in machine learning applications for similarity search and recommendation systems.",
            "metadata": {"topic": "vector_database", "difficulty": "advanced"}
        },
        {
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It enhances language models by providing relevant context from external knowledge sources.",
            "metadata": {"topic": "rag", "difficulty": "intermediate"}
        },
        {
            "text": "Embeddings are numerical representations of text, images, or other data that capture semantic meaning. They enable machines to understand relationships between different pieces of information.",
            "metadata": {"topic": "embeddings", "difficulty": "intermediate"}
        },
        {
            "text": "Hybrid search combines multiple search techniques, such as keyword-based search and vector similarity search, to provide more accurate and relevant results. It leverages the strengths of different search approaches.",
            "metadata": {"topic": "search", "difficulty": "advanced"}
        }
    ]

def print_search_results(results, search_type):
    """Print search results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{search_type.upper()} RESULTS")
    print(f"{'='*60}")
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Document ID: {result['id']}")
        print(f"   Content: {result['content'][:150]}...")
        
        if 'metadata' in result and result['metadata']:
            print(f"   Metadata: {json.dumps(result['metadata'], indent=2)}")
        
        # Print scores based on search type
        if 'combined_score' in result:
            print(f"   Combined Score: {result['combined_score']:.4f}")
            if 'vector_distance' in result and result['vector_distance'] is not None:
                print(f"   Vector Distance: {result['vector_distance']:.4f}")
            if 'text_score' in result and result['text_score'] is not None:
                print(f"   Text Score: {result['text_score']:.4f}")
        elif 'similarity' in result:
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Distance: {result['distance']:.4f}")
        elif 'score' in result:
            print(f"   Relevance Score: {result['score']:.4f}")
        
        print("-" * 60)

def main():
    """Main function to demonstrate the RAG pipeline."""
    print("🚀 Starting RAG Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize the RAG pipeline
    with RAGPipeline() as rag:
        print("\n📊 Initial document count:", rag.get_document_count())
        
        # Ingest sample documents
        print("\n📥 Ingesting sample documents...")
        sample_docs = create_sample_documents()
        doc_ids = rag.ingest_documents(sample_docs)
        print(f"✅ Successfully ingested {len(doc_ids)} documents")
        print(f"📊 Total documents in database: {rag.get_document_count()}")
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "neural networks and deep learning",
            "database systems and PostgreSQL",
            "vector similarity search",
            "natural language processing techniques"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Hybrid search
            hybrid_results = rag.search(query, k=3, hybrid_weight=0.5)
            print_search_results(hybrid_results, "Hybrid Search")
            
            # Vector search only
            vector_results = rag.vector_search(query, k=3)
            print_search_results(vector_results, "Vector Search")
            
            # Full-text search only
            text_results = rag.full_text_search(query, k=3)
            print_search_results(text_results, "Full-Text Search")
        
        # Test different hybrid weights
        print(f"\n🔍 Testing different hybrid weights for query: 'machine learning'")
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            results = rag.search("machine learning", k=2, hybrid_weight=weight)
            print(f"\nHybrid Search (weight={weight}):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['combined_score']:.4f} - {result['content'][:80]}...")
        
        # Retrieve a specific document
        if doc_ids:
            print(f"\n📄 Retrieving specific document (ID: {doc_ids[0]})")
            doc = rag.get_document_by_id(doc_ids[0])
            if doc:
                print(f"Document ID: {doc['id']}")
                print(f"Content: {doc['content'][:200]}...")
                print(f"Metadata: {json.dumps(doc['metadata'], indent=2)}")
                print(f"Created at: {doc['created_at']}")
    
    print("\n✅ RAG Pipeline demonstration completed successfully!")

if __name__ == "__main__":
    main()

