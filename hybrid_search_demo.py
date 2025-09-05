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
        print("using enhanced hybrid search (BM25 + pgvector) by default.")
        print("\nCommands:")
        print("  - Type your search query to find relevant documents (uses hybrid search)")
        print("  - Type 'count' to see total documents in the database")
        print("  - Type 'ingest' to load markdown files from the data directory")
        print("  - Type 'bm25 <query>' for pure BM25 keyword search")
        print("  - Type 'vector <query>' for pure vector semantic search")
        print("  - Type 'enhanced <query>' for advanced hybrid search with options")
        print("  - Type 'stats' to see BM25 index statistics")
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
                score = result.get('score', result.get('similarity', result.get('combined_score', 'N/A')))
                if isinstance(score, (int, float)):
                    print(f"   📊 Score: {score:.4f}")
                else:
                    print(f"   📊 Score: {score}")

            print("-" * 40)

    def handle_search(self, query: str):
        """Handle search queries."""
        try:
            print(f"\n🔍 Searching for: '{query}'")

            # Perform enhanced hybrid search (now the default)
            results = self.retrieval.search(query, k=Config.DEFAULT_K, hybrid_weight=Config.DEFAULT_HYBRID_WEIGHT)

            if results:
                # Show fusion method and weights used
                fusion_method = results[0].get('fusion_method', Config.DEFAULT_FUSION_METHOD)
                vector_weight = results[0].get('vector_weight', Config.DEFAULT_HYBRID_WEIGHT)
                query_aware = results[0].get('query_aware', True)
                print(f"   Using {fusion_method} fusion with {vector_weight:.2f} vector weight (query-aware: {query_aware})")
                
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

    def handle_bm25_search(self, query: str):
        """Handle BM25 keyword search."""
        try:
            print(f"\n🔍 BM25 Keyword Search for: '{query}'")
            results = self.retrieval.bm25_search(query, k=Config.DEFAULT_K)
            
            if results:
                self.display_search_results(results, query)
            else:
                print(f"\n❌ No results found for: '{query}'")
                
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            print(f"\n❌ BM25 search failed: {e}")

    def handle_vector_search(self, query: str):
        """Handle vector semantic search."""
        try:
            print(f"\n🔍 Vector Semantic Search for: '{query}'")
            results = self.retrieval.vector_search(query, k=Config.DEFAULT_K)
            
            if results:
                self.display_search_results(results, query)
            else:
                print(f"\n❌ No results found for: '{query}'")
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            print(f"\n❌ Vector search failed: {e}")

    def handle_enhanced_search(self, query: str):
        """Handle enhanced hybrid search."""
        try:
            print(f"\n🔍 Enhanced Hybrid Search for: '{query}'")
            results = self.retrieval.enhanced_hybrid_search(query, k=Config.DEFAULT_K)
            
            if results:
                # Show fusion method and weights used
                if results:
                    fusion_method = results[0].get('fusion_method', Config.DEFAULT_FUSION_METHOD)
                    vector_weight = results[0].get('vector_weight', Config.DEFAULT_HYBRID_WEIGHT)
                    print(f"   Using {fusion_method} fusion with {vector_weight:.2f} vector weight")
                
                self.display_search_results(results, query)
            else:
                print(f"\n❌ No results found for: '{query}'")
                
        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            print(f"\n❌ Enhanced search failed: {e}")

    def handle_stats(self):
        """Handle BM25 statistics request."""
        try:
            stats = self.retrieval.get_bm25_stats()
            print(f"\n📊 BM25 Index Statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Unique terms: {stats['unique_terms']}")
            print(f"   Average doc length: {stats['average_doc_length']:.1f} terms")
            print(f"   BM25 k1 parameter: {stats['k1_parameter']}")
            print(f"   BM25 b parameter: {stats['b_parameter']}")
        except Exception as e:
            logger.error(f"Stats error: {e}")
            print(f"\n❌ Failed to get statistics: {e}")

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

                elif user_input.lower() == "stats":
                    self.handle_stats()

                elif user_input.lower().startswith("bm25 "):
                    query = user_input[5:].strip()
                    if query:
                        self.handle_bm25_search(query)
                    else:
                        print("❌ Please provide a query after 'bm25'")

                elif user_input.lower().startswith("vector "):
                    query = user_input[7:].strip()
                    if query:
                        self.handle_vector_search(query)
                    else:
                        print("❌ Please provide a query after 'vector'")

                elif user_input.lower().startswith("enhanced "):
                    query = user_input[9:].strip()
                    if query:
                        self.handle_enhanced_search(query)
                    else:
                        print("❌ Please provide a query after 'enhanced'")

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
