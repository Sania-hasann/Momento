-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table with vector column (matching the Python code expectations)
CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(384),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents 
USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_documents_content 
ON documents 
USING gin(to_tsvector('english', content));