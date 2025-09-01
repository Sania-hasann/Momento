#!/usr/bin/env python3
"""
Hybrid Search Demo - Interactive document search using vector similarity and full-text search.
"""

import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modules
from retrieval import Retrieval
from document_ingestion import DocumentIngestion
from config import Config


class HybridSearchDemo:
    """
    Interactive demo for hybrid search capabilities.
    """

    def __init__(self):
        """Initialize the hybrid search demo."""
        try:
            self.retrieval = Retrieval()
            self.ingestion = DocumentIngestion()
            logger.info("Hybrid Search Demo initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize demo: {e}")
            sys.exit(1)

    def display_welcome(self):
        """Display welcome message and instructions."""
        print("\n" + "=" * 60)
        print("🔍 HYBRID SEARCH DEMO")
        print("=" * 60)
        print("This demo allows you to search through your document collection")
        print("using both vector similarity and full-text search capabilities.")
        print("\nCommands:")
        print("  - Type your search query to find relevant documents")
        print("  - Type 'count' to see total documents in the database")
        print("  - Type 'ingest' to load markdown files from the data directory")
        print("  - Type 'quit' or 'exit' to end the demo")
        print("=" * 60)

    def display_search_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results in a formatted way."""
        if not results:
            print(f"\n❌ No results found for query: '{query}'")
            return

        print(f"\n📋 Found {len(results)} relevant documents for: '{query}'")
        print("-" * 60)

        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i}:")

            # Extract content (truncate if too long)
            content = result.get("content", "No content available")
            if len(content) > 300:
                content = content[:300] + "..."

            print(f"   Content: {content}")

            # Display metadata
            metadata = result.get("metadata", {})
            if metadata:
                print(f"   📁 File: {metadata.get('file_name', 'Unknown')}")
                print(f"   📂 Directory: {metadata.get('directory', 'Unknown')}")
                print(f"   📊 Score: {result.get('score', 'N/A'):.4f}")

            print("-" * 40)

    def handle_search(self, query: str):
        """Handle search queries."""
        try:
            print(f"\n🔍 Searching for: '{query}'")

            # Perform hybrid search
            results = self.retrieval.search(query, k=5, hybrid_weight=0.5)

            if results:
                self.display_search_results(results, query)
            else:
                print(f"\n❌ No results found for: '{query}'")

        except Exception as e:
            logger.error(f"Search error: {e}")
            print(f"\n❌ Search failed: {e}")

    def handle_count(self):
        """Handle document count request."""
        try:
            count = self.ingestion.get_document_count()
            print(f"\n📊 Total documents in database: {count}")
        except Exception as e:
            logger.error(f"Count error: {e}")
            print(f"\n❌ Failed to get document count: {e}")

    def handle_ingest(self):
        """Handle document ingestion request."""
        data_dir = "data"

        if not os.path.exists(data_dir):
            print(f"\n❌ Data directory '{data_dir}' not found!")
            return

        try:
            print(f"\n📚 Ingesting markdown files from '{data_dir}'...")
            doc_ids = self.ingestion.ingest_markdown_directory(data_dir, recursive=True)
            print(f"✅ Successfully ingested {len(doc_ids)} documents")

            # Show updated count
            self.handle_count()

        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            print(f"\n❌ Ingestion failed: {e}")

    def run(self):
        """Run the interactive demo."""
        self.display_welcome()

        while True:
            try:
                # Get user input
                user_input = input(
                    "\n🔍 Enter your search query (or command): "
                ).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                elif user_input.lower() == "count":
                    self.handle_count()

                elif user_input.lower() == "ingest":
                    self.handle_ingest()

                elif user_input.lower() == "help":
                    self.display_welcome()

                else:
                    # Treat as search query
                    self.handle_search(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Demo interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\n❌ An error occurred: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.retrieval.close()
            self.ingestion.close()
            logger.info("Demo cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function to run the hybrid search demo."""
    demo = None
    try:
        demo = HybridSearchDemo()
        demo.run()
    except Exception as e:
        logger.error(f"Demo failed to start: {e}")
        print(f"❌ Failed to start demo: {e}")
        sys.exit(1)
    finally:
        if demo:
            demo.cleanup()


if __name__ == "__main__":
    main()
