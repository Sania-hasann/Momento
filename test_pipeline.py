#!/usr/bin/env python3
"""
Simple test script for the RAG pipeline.

This script performs basic functionality tests to ensure the pipeline
is working correctly.
"""

import logging
import sys
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("🧪 Testing RAG Pipeline Basic Functionality")
    print("=" * 50)

    try:
        with RAGPipeline() as rag:
            # Test 1: Check initial document count
            initial_count = rag.get_document_count()
            print(f"✅ Initial document count: {initial_count}")

            # Test 2: Ingest a simple document
            test_text = "This is a test document about artificial intelligence and machine learning."
            doc_id = rag.ingest_text(test_text, {"test": True, "topic": "ai"})
            print(f"✅ Ingested test document with ID: {doc_id}")

            # Test 3: Check document count after ingestion
            new_count = rag.get_document_count()
            print(f"✅ Document count after ingestion: {new_count}")

            # Test 4: Retrieve the document by ID
            retrieved_doc = rag.get_document_by_id(doc_id)
            if retrieved_doc and retrieved_doc["content"] == test_text:
                print("✅ Document retrieval by ID successful")
            else:
                print("❌ Document retrieval by ID failed")
                return False

            # Test 5: Perform a search
            results = rag.search("artificial intelligence", k=1)
            if results and len(results) > 0:
                print("✅ Search functionality working")
                print(f"   Found {len(results)} results")
            else:
                print("❌ Search functionality failed")
                return False

            # Test 6: Test vector search
            vector_results = rag.vector_search("machine learning", k=1)
            if vector_results and len(vector_results) > 0:
                print("✅ Vector search working")
            else:
                print("❌ Vector search failed")
                return False

            # Test 7: Test full-text search
            text_results = rag.full_text_search("test document", k=1)
            if text_results and len(text_results) > 0:
                print("✅ Full-text search working")
            else:
                print("❌ Full-text search failed")
                return False

            print("\n🎉 All basic functionality tests passed!")
            return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False


def test_hybrid_search():
    """Test hybrid search with different weights."""
    print("\n🧪 Testing Hybrid Search")
    print("=" * 30)

    try:
        with RAGPipeline() as rag:
            # Ingest multiple test documents
            test_docs = [
                "Machine learning algorithms are used for pattern recognition.",
                "Deep learning is a subset of machine learning using neural networks.",
                "Natural language processing helps computers understand human language.",
                "PostgreSQL is a powerful relational database system.",
            ]

            doc_ids = []
            for i, doc in enumerate(test_docs):
                doc_id = rag.ingest_text(doc, {"test": True, "index": i})
                doc_ids.append(doc_id)

            print(f"✅ Ingested {len(doc_ids)} test documents")

            # Test different hybrid weights
            query = "machine learning"
            weights = [0.0, 0.5, 1.0]

            for weight in weights:
                results = rag.search(query, k=2, hybrid_weight=weight)
                print(f"   Weight {weight}: Found {len(results)} results")

                if results:
                    top_score = results[0]["combined_score"]
                    print(f"      Top score: {top_score:.4f}")

            print("✅ Hybrid search tests completed")
            return True

    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting RAG Pipeline Tests")
    print("=" * 60)

    # Run basic functionality tests
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        sys.exit(1)

    # Run hybrid search tests
    if not test_hybrid_search():
        print("\n❌ Hybrid search tests failed")
        sys.exit(1)

    print("\n🎉 All tests passed successfully!")
    print("\n📝 The RAG pipeline is working correctly.")
    print("   You can now run 'python example.py' for a full demonstration.")


if __name__ == "__main__":
    main()
