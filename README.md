# RAG Pipeline with PostgreSQL and pgvector

A comprehensive Retrieval-Augmented Generation (RAG) pipeline that combines document ingestion and hybrid search capabilities using PostgreSQL with pgvector for efficient vector similarity search.

## Features

- **Document Ingestion**: Process raw text documents or files with automatic embedding generation
- **Custom Splitters**: Intelligent text splitting with specialized handlers for markdown and Unicode content
- **Hybrid Search**: Combine full-text search and vector similarity search for optimal results
- **PostgreSQL Integration**: Use PostgreSQL with pgvector extension for robust vector storage
- **all-MiniLM-L6-v2 Model**: Fast and efficient sentence embeddings (384 dimensions)
- **Flexible Search Options**: Support for vector-only, full-text-only, and hybrid search
- **Metadata Support**: Store and query document metadata alongside content
- **Production Ready**: Well-structured code with proper error handling and logging

## Custom Splitters

The pipeline includes intelligent text splitters that automatically choose the best splitting strategy based on content type:

### MarkdownTokenTextSplitter
- **Purpose**: Optimized for markdown documents
- **Features**:
  - Preserves markdown structure (headers, tables, lists)
  - Keeps tables together as complete units
  - Maintains header context across chunks
  - Handles code blocks and formatting
- **Use Cases**: Documentation, technical content, structured markdown files

### UnicodeSafeTokenTextSplitter
- **Purpose**: Safe handling of Unicode and international text (DEFAULT)
- **Features**:
  - Preserves surrogate pairs and complex Unicode characters
  - Handles multi-language content (Chinese, Japanese, Arabic, etc.)
  - Safe tokenization for emoji and special characters
  - Used as the default splitter for all documents
- **Use Cases**: All content types, international content, multilingual documents, content with emojis

### SentenceSplitter (Legacy)
- **Purpose**: General-purpose text splitting (legacy option)
- **Features**:
  - Simple sentence-based chunking
  - Configurable chunk size and overlap
  - Efficient for plain text content
- **Use Cases**: Legacy compatibility, simple content (when explicitly specified)

The system automatically detects content type and applies the appropriate splitter, ensuring optimal document processing for different content types. **UnicodeSafeTokenTextSplitter is now the default for all documents** to ensure safe handling of any Unicode characters that might be present.

## Architecture

The pipeline consists of two main classes:

### DocumentIngestion
- Accepts raw text documents or file paths
- Automatically selects appropriate custom splitter based on content
- Uses UnicodeSafeTokenTextSplitter as the default for all documents
- Generates embeddings using all-MiniLM-L6-v2 model
- Stores documents and embeddings in PostgreSQL with pgvector
- Supports batch ingestion and metadata storage
- Tracks which splitter was used for each document

### Retrieval
- Performs hybrid search combining full-text and vector similarity
- Uses consistent splitters for query processing
- Supports configurable search weights for different search types
- Provides individual vector and full-text search capabilities
- Returns ranked results with detailed scoring information

## Prerequisites

1. **PostgreSQL 12+** with pgvector extension
2. **Python 3.8+**
3. **pgvector extension** installed in PostgreSQL

### Installing pgvector

```bash
# On Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# On macOS with Homebrew
brew install pgvector

# Or build from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RAG
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_database
DB_USER=your_username
DB_PASSWORD=your_password
VECTOR_DIMENSION=384
```

4. **Create the database**:
```sql
CREATE DATABASE rag_database;
```

## Usage

### Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
with RAGPipeline() as rag:
    # Ingest documents
    doc_id = rag.ingest_text(
        "Machine learning is a subset of artificial intelligence.",
        metadata={"topic": "ai", "difficulty": "beginner"}
    )
    
    # Perform hybrid search
    results = rag.search("machine learning algorithms", k=5)
    
    # Print results
    for result in results:
        print(f"Score: {result['combined_score']:.4f}")
        print(f"Content: {result['content'][:100]}...")
```

### Custom Splitter Usage

The pipeline automatically selects the best splitter based on content type:

```python
from document_ingestion import DocumentIngestion

ingestion = DocumentIngestion()

# Markdown content - uses MarkdownTokenTextSplitter
markdown_content = """
# AI and Machine Learning

## Introduction
Machine learning is a subset of artificial intelligence.

### Key Concepts
- **Supervised Learning**: Learning from labeled data
- **Unsupervised Learning**: Finding patterns in unlabeled data

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised | Prediction |
| K-means | Unsupervised | Clustering |
"""

markdown_id = ingestion.ingest_text(markdown_content, {
    'content_type': 'markdown',
    'topic': 'machine_learning'
})

# Unicode content - uses UnicodeSafeTokenTextSplitter
unicode_content = """
International AI Research: 人工智能研究

Key terms:
- 机器学习 (Machine Learning)
- 深度学习 (Deep Learning)
- 神经网络 (Neural Networks)

Emoji examples: 🧠 🤖 📊
"""

unicode_id = ingestion.ingest_text(unicode_content, {
    'content_type': 'unicode',
    'language': 'multilingual'
})

# Regular text - uses UnicodeSafeTokenTextSplitter (default)
regular_text = """
This is a simple text document about artificial intelligence.
It contains basic sentences without special formatting.
The system will use the UnicodeSafeTokenTextSplitter as the default.
"""

regular_id = ingestion.ingest_text(regular_text, {
    'content_type': 'regular',
    'topic': 'ai_basics'
})

# Check which splitters were used
stats = ingestion.get_splitter_usage_stats()
print(f"Splitter usage: {stats}")

ingestion.close()
```

### Testing Custom Splitters

Run the test script to see the custom splitters in action:

```bash
python test_custom_splitters.py
```

This will:
1. Test different content types (markdown, Unicode, regular text)
2. Show which splitter is automatically selected
3. Display splitter usage statistics
4. Test with actual markdown files from the data directory

### Advanced Usage

```python
from rag_pipeline import RAGPipeline

with RAGPipeline() as rag:
    # Ingest multiple documents
    documents = [
        {"text": "Document 1 content", "metadata": {"source": "book1"}},
        {"text": "Document 2 content", "metadata": {"source": "book2"}},
        {"file_path": "path/to/document.txt", "metadata": {"source": "file"}}
    ]
    
    doc_ids = rag.ingest_documents(documents)
    
    # Different search types
    # Hybrid search (default)
    hybrid_results = rag.search("query", k=5, hybrid_weight=0.5)
    
    # Vector search only
    vector_results = rag.vector_search("query", k=5)
    
    # Full-text search only
    text_results = rag.full_text_search("query", k=5)
    
    # Retrieve specific document
    doc = rag.get_document_by_id(doc_ids[0])
```

### Running the Example

```bash
python example.py
```

This will:
1. Create sample documents about AI/ML topics
2. Ingest them into the database
3. Perform various search queries
4. Demonstrate different search types and weights
5. Show detailed results with scoring

## Database Schema

The pipeline creates a `documents` table with the following structure:

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

- **Vector index**: `ivfflat` index on embedding column for fast similarity search
- **Full-text index**: `gin` index on content for efficient text search

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `rag_database` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `` |
| `VECTOR_DIMENSION` | Embedding dimension | `384` |

### Search Parameters

- **k**: Number of results to return (default: 5)
- **hybrid_weight**: Weight for vector similarity in hybrid search (0.0-1.0, default: 0.5)
  - 0.0 = full-text search only
  - 1.0 = vector search only
  - 0.5 = equal weight for both

## Performance Considerations

1. **Vector Index**: The `ivfflat` index provides fast approximate similarity search
2. **Batch Operations**: Use `ingest_documents()` for bulk ingestion
3. **Connection Pooling**: The pipeline manages database connections efficiently
4. **Memory Usage**: all-MiniLM-L6-v2 is optimized for speed and memory usage

## Error Handling

The pipeline includes comprehensive error handling:
- Database connection errors
- File not found errors
- Embedding generation errors
- Invalid search parameters

All errors are logged with appropriate context and re-raised for handling by the calling code.

## Contributing

1. Follow the existing code structure and patterns
2. Add proper docstrings for all functions
3. Include error handling and logging
4. Update tests when adding new features
5. Follow SOLID principles for Python development

## License

This project is licensed under the MIT License - see the LICENSE file for details.

