#!/usr/bin/env python3
"""
Test script to verify the document ingestion fix.
"""

import logging
from document_ingestion import DocumentIngestion

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_document_ingestion():
    """Test that document ingestion works correctly."""

    # Initialize the ingestion pipeline
    ingestion = DocumentIngestion()

    # Test with a simple markdown content
    test_content = """
# Test Document

This is a test document to verify that the ingestion pipeline works correctly.

## Features
- **Bold text**
- *Italic text*
- `Code snippets`

## Table Example
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |

This should be processed without any errors.
"""

    try:
        logger.info("Testing document ingestion...")

        # Ingest the test content
        doc_id = ingestion.ingest_text(
            test_content,
            {"content_type": "test", "splitter_used": "MarkdownTokenTextSplitter"},
        )

        logger.info(f"Successfully ingested document with ID: {doc_id}")

        # Get document count
        count = ingestion.get_document_count()
        logger.info(f"Total documents in database: {count}")

        # Get splitter usage stats
        stats = ingestion.get_splitter_usage_stats()
        logger.info(f"Splitter usage stats: {stats}")

        logger.info("Document ingestion test completed successfully!")

    except Exception as e:
        logger.error(f"Error during document ingestion test: {e}")
        raise
    finally:
        ingestion.close()


if __name__ == "__main__":
    logger.info("Starting document ingestion fix test...")
    test_document_ingestion()
    logger.info("Test completed!")
