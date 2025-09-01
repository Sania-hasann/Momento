# Momento RAG Pipeline - Setup Guide

## Overview
This is a hybrid search pipeline using PostgreSQL with pgvector for vector similarity search and full-text search capabilities. The system can ingest markdown documents and provide intelligent search results.

## Recent Fixes Applied

### 1. Environment Variable Discrepancies Fixed
- **Database Name**: Aligned Docker Compose to use `rag_database` (was `vectordb`)
- **Table Name**: Updated `init.sql` to create `documents` table (was `items`)
- **Environment Variables**: Created `.env` file with correct defaults
- **Docker Compose**: Updated to use environment variables with proper defaults

### 2. Document Ingestion Enhanced
- **Recursive Markdown Processing**: Added `ingest_markdown_directory()` method
- **Metadata Tracking**: Captures file paths, directories, and source information
- **Error Handling**: Improved error handling for file processing

### 3. Hybrid Search Demo Created
- **Interactive Interface**: Command-line demo with search capabilities
- **Real-time Search**: Query documents using hybrid search
- **Document Management**: Count documents and ingest new files

## Quick Start

### 1. Start the Database
```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Check if it's running
docker compose ps
```

### 2. Test the Setup
```bash
# Run the test script to verify everything works
python test_setup.py
```

### 3. Run the Hybrid Search Demo
```bash
# Start the interactive demo
python hybrid_search_demo.py
```

## Demo Commands

Once the demo is running, you can use these commands:

- **Search**: Type any query to search documents
- **`ingest`**: Load all markdown files from the `data/` directory
- **`count`**: Show total number of documents in the database
- **`help`**: Show available commands
- **`quit`** or **`exit`**: End the demo

## Example Usage

```bash
# Start the demo
python hybrid_search_demo.py

# In the demo:
🔍 Enter your search query (or command): ingest
📚 Ingesting markdown files from 'data'...
✅ Successfully ingested 45 documents

🔍 Enter your search query (or command): debit card
📋 Found 5 relevant documents for: 'debit card'
📄 Result 1:
   Content: UBL Signature Debit MasterCard offers premium banking...
   📁 File: UBL-Signature-Debit-MasterCard.md
   📂 Directory: Consumer_Banking/Card_Products/UBL_Debit_Card
   📊 Score: 0.8923
```

## Configuration

### Environment Variables (.env)
```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_database
DB_USER=postgres
DB_PASSWORD=password

# Vector Settings
VECTOR_DIMENSION=384

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Database Schema
The system uses a `documents` table with:
- `id`: Primary key
- `content`: Document text content
- `embedding`: Vector representation (384 dimensions)
- `metadata`: JSON metadata (file info, paths, etc.)
- `created_at`: Timestamp

## Architecture

### Components
1. **Document Ingestion**: Processes markdown files recursively
2. **Vector Store**: PostgreSQL with pgvector extension
3. **Hybrid Search**: Combines vector similarity and full-text search
4. **Interactive Demo**: Command-line interface for testing

### Search Capabilities
- **Vector Similarity**: Semantic search using embeddings
- **Full-text Search**: Traditional keyword matching
- **Hybrid Search**: Weighted combination of both approaches
- **Metadata Filtering**: Search by file paths, directories, etc.

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   docker compose ps
   
   # Check logs
   docker compose logs postgres
   ```

2. **No Documents Found**
   ```bash
   # Use 'ingest' command in the demo to load files
   # Or check if data directory has markdown files
   find data -name "*.md" | wc -l
   ```

3. **Import Errors**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

### Logs
- Application logs are displayed in the console
- Database logs: `docker compose logs postgres`
- PgAdmin logs: `docker compose logs pgadmin`

## Development

### Adding New Document Types
1. Modify `ingest_markdown_directory()` in `document_ingestion.py`
2. Add new file extensions to `markdown_extensions` set
3. Update metadata handling as needed

### Customizing Search
1. Modify `hybrid_weight` parameter in `hybrid_search_demo.py`
2. Adjust `k` parameter for number of results
3. Customize result display in `display_search_results()`

### Database Management
- **PgAdmin**: Access at http://localhost:5050
  - Email: admin@example.com
  - Password: admin
- **Direct Connection**: localhost:5432
  - Database: rag_database
  - User: postgres
  - Password: password

## Performance Tips

1. **Index Optimization**: The system creates optimized indexes for both vector and text search
2. **Chunking**: Documents are split into 512-character chunks with 50-character overlap
3. **Batch Processing**: Multiple documents can be ingested simultaneously
4. **Connection Pooling**: Database connections are managed efficiently

## Security Notes

- Default passwords are used for development
- Change passwords in production
- Consider using environment variables for sensitive data
- PgAdmin should be disabled in production environments 