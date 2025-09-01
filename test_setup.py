#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        from config import Config

        print("✅ Config imported successfully")

        from database import DatabaseManager

        print("✅ DatabaseManager imported successfully")

        from document_ingestion import DocumentIngestion

        print("✅ DocumentIngestion imported successfully")

        from retrieval import Retrieval

        print("✅ Retrieval imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")

    try:
        from config import Config

        print(f"   DB_HOST: {Config.DB_HOST}")
        print(f"   DB_PORT: {Config.DB_PORT}")
        print(f"   DB_NAME: {Config.DB_NAME}")
        print(f"   DB_USER: {Config.DB_USER}")
        print(f"   VECTOR_DIMENSION: {Config.VECTOR_DIMENSION}")
        print(f"   EMBEDDING_MODEL: {Config.EMBEDDING_MODEL}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_database_connection():
    """Test database connection."""
    print("\n🔍 Testing database connection...")

    try:
        from database import DatabaseManager

        with DatabaseManager() as db:
            conn = db.get_connection()
            print("✅ Database connection successful")

            # Test if pgvector extension is available
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if cursor.fetchone():
                    print("✅ pgvector extension is active")
                else:
                    print("❌ pgvector extension not found")
                    return False

            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_document_ingestion():
    """Test document ingestion initialization."""
    print("\n🔍 Testing document ingestion...")

    try:
        from document_ingestion import DocumentIngestion

        ingestion = DocumentIngestion()
        print("✅ DocumentIngestion initialized successfully")

        count = ingestion.get_document_count()
        print(f"   Current document count: {count}")

        ingestion.close()
        return True
    except Exception as e:
        print(f"❌ Document ingestion test failed: {e}")
        return False


def test_retrieval():
    """Test retrieval initialization."""
    print("\n🔍 Testing retrieval...")

    try:
        from retrieval import Retrieval

        retrieval = Retrieval()
        print("✅ Retrieval initialized successfully")

        retrieval.close()
        return True
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Momento RAG Pipeline Setup")
    print("=" * 50)

    tests = [
        test_imports,
        test_config,
        test_database_connection,
        test_document_ingestion,
        test_retrieval,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n📝 Next steps:")
        print("   1. Start the database: docker compose up -d")
        print("   2. Run the demo: python hybrid_search_demo.py")
        print("   3. Use 'ingest' command to load your markdown files")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
