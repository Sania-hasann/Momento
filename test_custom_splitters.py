#!/usr/bin/env python3
"""
Test script to demonstrate the custom splitter integration in the ingestion pipeline.
"""

import logging
import os
from document_ingestion import DocumentIngestion

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_custom_splitters():
    """Test the custom splitter integration with different types of content."""

    # Initialize the ingestion pipeline
    ingestion = DocumentIngestion()

    # Test 1: Markdown content
    markdown_content = """
# Sample Markdown Document

## Introduction
This is a sample markdown document to test the custom splitter.

### Features
- **Bold text** and *italic text*
- [Links](https://example.com)
- `Inline code`

## Table Example
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

## Code Block
```python
def hello_world():
    print("Hello, World!")
```

## Numbered List
1. First item
2. Second item
3. Third item

This is a long paragraph that should be split appropriately by the custom markdown splitter. 
It contains multiple sentences and should be processed according to the markdown structure 
while preserving headers, tables, and other formatting elements.
"""

    # Test 2: Unicode content
    unicode_content = """
This is a document with Unicode characters: 
- Café (with accent)
- 你好 (Chinese characters)
- Привет (Cyrillic characters)
- 🌟 Emoji characters
- こんにちは (Japanese characters)

This content should be handled by the UnicodeSafeTokenTextSplitter to ensure proper 
tokenization of Unicode characters including surrogate pairs.
"""

    # Test 3: Regular text
    regular_text = """
This is a regular text document without any special formatting.
It contains simple sentences and paragraphs that should be processed
by the default sentence splitter.

The content is straightforward and doesn't require any special handling
for markdown or Unicode characters.
"""

    try:
        logger.info("Testing custom splitter integration...")

        # Ingest different types of content
        logger.info("Ingesting markdown content...")
        markdown_id = ingestion.ingest_text(
            markdown_content,
            {"content_type": "markdown", "splitter_used": "MarkdownTokenTextSplitter"},
        )

        logger.info("Ingesting Unicode content...")
        unicode_id = ingestion.ingest_text(
            unicode_content,
            {
                "content_type": "unicode",
                "splitter_used": "UnicodeSafeTokenTextSplitter",
            },
        )

        logger.info("Ingesting regular text...")
        regular_id = ingestion.ingest_text(
            regular_text,
            {
                "content_type": "regular",
                "splitter_used": "UnicodeSafeTokenTextSplitter",  # Now the default for regular text
            },
        )

        # Get splitter usage statistics
        logger.info("Getting splitter usage statistics...")
        stats = ingestion.get_splitter_usage_stats()
        logger.info(f"Splitter usage stats: {stats}")

        # Get total document count
        total_count = ingestion.get_document_count()
        logger.info(f"Total documents in database: {total_count}")

        logger.info("Custom splitter integration test completed successfully!")
        logger.info(
            f"Document IDs: Markdown={markdown_id}, Unicode={unicode_id}, Regular={regular_id}"
        )

    except Exception as e:
        logger.error(f"Error during custom splitter test: {e}")
        raise
    finally:
        ingestion.close()


def test_markdown_file_ingestion():
    """Test ingesting actual markdown files from the data directory."""

    # Check if data directory exists
    if not os.path.exists("data"):
        logger.warning(
            "Data directory not found. Skipping markdown file ingestion test."
        )
        return

    ingestion = DocumentIngestion()

    try:
        logger.info("Testing markdown file ingestion...")

        # Ingest markdown files from the data directory
        doc_ids = ingestion.ingest_markdown_directory("data", recursive=True)

        logger.info(f"Successfully ingested {len(doc_ids)} markdown files")

        # Get splitter usage statistics
        stats = ingestion.get_splitter_usage_stats()
        logger.info(f"Splitter usage stats after file ingestion: {stats}")

    except Exception as e:
        logger.error(f"Error during markdown file ingestion test: {e}")
        raise
    finally:
        ingestion.close()


if __name__ == "__main__":
    logger.info("Starting custom splitter integration tests...")

    # Test with sample content
    test_custom_splitters()

    # Test with actual markdown files
    test_markdown_file_ingestion()

    logger.info("All tests completed!")
