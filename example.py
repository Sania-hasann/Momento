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
import os
from datetime import datetime
from rag_pipeline import RAGPipeline

def setup_logging():
    """Set up logging configuration to write to logs.txt file."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"logs_{timestamp}.txt")
    
    # Configure logging with a cleaner format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()  # Keep console output for immediate feedback
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

def create_sample_documents():
    """Create diverse sample documents for comprehensive testing."""
    logger.info("Creating diverse sample documents for comprehensive testing")
    return [
        # Machine Learning & AI
        {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns. Supervised learning, unsupervised learning, and reinforcement learning are the three main categories of machine learning approaches.",
            "metadata": {"topic": "machine_learning", "difficulty": "intermediate", "category": "ai_ml", "length": "long"}
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to model complex patterns. It excels in image recognition, NLP, and speech processing. Convolutional Neural Networks (CNNs) are particularly effective for computer vision tasks.",
            "metadata": {"topic": "deep_learning", "difficulty": "advanced", "category": "ai_ml", "length": "medium"}
        },
        {
            "text": "Natural Language Processing (NLP) enables machines to understand human language. It includes tasks like sentiment analysis, machine translation, and text generation. Modern NLP relies heavily on transformer models like BERT and GPT.",
            "metadata": {"topic": "nlp", "difficulty": "intermediate", "category": "ai_ml", "length": "medium"}
        },
        {
            "text": "AI ethics focuses on ensuring artificial intelligence systems are fair, transparent, and accountable. Key concerns include bias mitigation, privacy protection, and explainable AI.",
            "metadata": {"topic": "ai_ethics", "difficulty": "intermediate", "category": "ai_ml", "length": "short"}
        },
        
        # Database Systems
        {
            "text": "PostgreSQL is a powerful, open-source object-relational database system. It extends SQL with advanced features including JSON support, arrays, custom types, and full-text search capabilities. PostgreSQL is known for its ACID compliance and extensibility.",
            "metadata": {"topic": "postgresql", "difficulty": "intermediate", "category": "databases", "length": "long"}
        },
        {
            "text": "MongoDB is a NoSQL document database that stores data in flexible, JSON-like documents. It's designed for scalability and high availability, making it popular for web applications and big data projects.",
            "metadata": {"topic": "mongodb", "difficulty": "beginner", "category": "databases", "length": "medium"}
        },
        {
            "text": "Vector databases store high-dimensional vectors for similarity search. They're essential for AI applications like recommendation systems, image search, and semantic text search. Popular options include Pinecone, Weaviate, and pgvector.",
            "metadata": {"topic": "vector_databases", "difficulty": "advanced", "category": "databases", "length": "medium"}
        },
        {
            "text": "Database indexing improves query performance by creating data structures that allow faster data retrieval. B-tree indexes are most common, while hash indexes work well for equality comparisons.",
            "metadata": {"topic": "database_indexing", "difficulty": "intermediate", "category": "databases", "length": "short"}
        },
        
        # RAG & Search
        {
            "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It enhances language models by providing relevant context from external knowledge sources, improving accuracy and reducing hallucinations.",
            "metadata": {"topic": "rag", "difficulty": "intermediate", "category": "search", "length": "medium"}
        },
        {
            "text": "Embeddings are numerical representations that capture semantic meaning. They enable similarity calculations between text, images, or other data types. Popular embedding models include Word2Vec, BERT, and sentence-transformers.",
            "metadata": {"topic": "embeddings", "difficulty": "intermediate", "category": "search", "length": "medium"}
        },
        {
            "text": "Hybrid search combines multiple search techniques for better results. It typically merges keyword-based search with vector similarity search, leveraging the strengths of both approaches.",
            "metadata": {"topic": "hybrid_search", "difficulty": "advanced", "category": "search", "length": "short"}
        },
        {
            "text": "Semantic search understands the meaning behind queries rather than just matching keywords. It uses embeddings and vector similarity to find conceptually related content.",
            "metadata": {"topic": "semantic_search", "difficulty": "intermediate", "category": "search", "length": "short"}
        },
        
        # Programming & Development
        {
            "text": "Python is a versatile programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and AI. Python's extensive library ecosystem makes it ideal for rapid prototyping.",
            "metadata": {"topic": "python", "difficulty": "beginner", "category": "programming", "length": "long"}
        },
        {
            "text": "Docker containers package applications with their dependencies. They provide consistent environments across different systems and enable microservices architecture. Docker Compose simplifies multi-container applications.",
            "metadata": {"topic": "docker", "difficulty": "intermediate", "category": "programming", "length": "medium"}
        },
        {
            "text": "REST APIs use HTTP methods for data exchange. They follow stateless principles and use JSON for data serialization. RESTful design emphasizes resource-based URLs and standard HTTP status codes.",
            "metadata": {"topic": "rest_apis", "difficulty": "beginner", "category": "programming", "length": "medium"}
        },
        {
            "text": "GraphQL provides a flexible alternative to REST APIs. It allows clients to request exactly the data they need, reducing over-fetching and under-fetching problems.",
            "metadata": {"topic": "graphql", "difficulty": "intermediate", "category": "programming", "length": "short"}
        },
        
        # Data Science & Analytics
        {
            "text": "Data preprocessing is crucial for machine learning success. It includes cleaning data, handling missing values, feature scaling, and encoding categorical variables. Proper preprocessing can significantly improve model performance.",
            "metadata": {"topic": "data_preprocessing", "difficulty": "intermediate", "category": "data_science", "length": "long"}
        },
        {
            "text": "Pandas is a powerful Python library for data manipulation and analysis. It provides DataFrame objects for handling structured data and includes tools for reading, writing, and transforming data.",
            "metadata": {"topic": "pandas", "difficulty": "beginner", "category": "data_science", "length": "medium"}
        },
        {
            "text": "Feature engineering creates new features from existing data to improve model performance. Techniques include polynomial features, interaction terms, and domain-specific transformations.",
            "metadata": {"topic": "feature_engineering", "difficulty": "advanced", "category": "data_science", "length": "short"}
        },
        {
            "text": "A/B testing compares two versions to determine which performs better. It's essential for data-driven decision making in product development and marketing campaigns.",
            "metadata": {"topic": "ab_testing", "difficulty": "intermediate", "category": "data_science", "length": "short"}
        },
        
        # Cloud & DevOps
        {
            "text": "AWS provides comprehensive cloud computing services including compute, storage, databases, and AI. Popular services include EC2, S3, Lambda, and SageMaker for machine learning workloads.",
            "metadata": {"topic": "aws", "difficulty": "intermediate", "category": "cloud_devops", "length": "medium"}
        },
        {
            "text": "Kubernetes orchestrates containerized applications across clusters. It provides automatic scaling, load balancing, and self-healing capabilities for microservices architectures.",
            "metadata": {"topic": "kubernetes", "difficulty": "advanced", "category": "cloud_devops", "length": "medium"}
        },
        {
            "text": "CI/CD pipelines automate software delivery processes. Continuous Integration builds and tests code changes, while Continuous Deployment automatically deploys to production environments.",
            "metadata": {"topic": "cicd", "difficulty": "intermediate", "category": "cloud_devops", "length": "short"}
        },
        {
            "text": "Infrastructure as Code (IaC) manages infrastructure through configuration files. Tools like Terraform and CloudFormation enable version-controlled, reproducible infrastructure deployment.",
            "metadata": {"topic": "iac", "difficulty": "intermediate", "category": "cloud_devops", "length": "short"}
        },
        
        # Security & Privacy
        {
            "text": "OAuth 2.0 is an authorization framework for secure API access. It enables third-party applications to access resources without sharing credentials, using access tokens instead.",
            "metadata": {"topic": "oauth", "difficulty": "intermediate", "category": "security", "length": "medium"}
        },
        {
            "text": "Encryption protects data confidentiality through mathematical algorithms. Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses public-private key pairs.",
            "metadata": {"topic": "encryption", "difficulty": "advanced", "category": "security", "length": "short"}
        },
        {
            "text": "GDPR compliance ensures data protection and privacy for EU citizens. It requires organizations to implement data minimization, consent management, and breach notification procedures.",
            "metadata": {"topic": "gdpr", "difficulty": "intermediate", "category": "security", "length": "short"}
        }
    ]

def analyze_search_results(results, search_type, query):
    """Analyze search results for issues and improvements."""
    logger.info(f"🔍 {search_type.upper()} ANALYSIS for: '{query}'")
    
    if not results:
        logger.info("   ❌ No results found")
        return
    
    # Count unique documents
    unique_contents = set()
    unique_metadata = set()
    unique_categories = set()
    score_range = []
    
    for result in results:
        content_preview = result['content'][:50]
        unique_contents.add(content_preview)
        unique_metadata.add(str(result.get('metadata', {}).get('topic', 'unknown')))
        unique_categories.add(str(result.get('metadata', {}).get('category', 'unknown')))
        
        # Get the most relevant score
        if 'combined_score' in result:
            score = result['combined_score']
        elif 'similarity' in result:
            score = result['similarity']
        elif 'score' in result:
            score = result['score']
        else:
            score = 0.0
        score_range.append(score)
    
    logger.info(f"   📊 Results Analysis:")
    logger.info(f"      Total results: {len(results)}")
    logger.info(f"      Unique documents: {len(unique_contents)}")
    logger.info(f"      Unique topics: {len(unique_metadata)}")
    logger.info(f"      Unique categories: {len(unique_categories)}")
    logger.info(f"      Score range: {min(score_range):.3f} - {max(score_range):.3f}")
    
    # Check for duplicates
    if len(unique_contents) < len(results):
        logger.warning(f"      ⚠️  DUPLICATE ISSUE: {len(results) - len(unique_contents)} duplicate results detected!")
    
    # Check score distribution
    if max(score_range) - min(score_range) < 0.1:
        logger.warning(f"      ⚠️  POOR SCORE DIFFERENTIATION: Score range too narrow!")
    
    # Check diversity
    if len(unique_categories) < 2:
        logger.warning(f"      ⚠️  LOW DIVERSITY: Only {len(unique_categories)} category found!")
    
    # Show top 3 unique results
    logger.info(f"   🏆 Top Results:")
    seen_contents = set()
    count = 0
    
    for i, result in enumerate(results, 1):
        content_preview = result['content'][:50]
        if content_preview not in seen_contents and count < 3:
            seen_contents.add(content_preview)
            count += 1
            
            topic = result.get('metadata', {}).get('topic', 'unknown')
            category = result.get('metadata', {}).get('category', 'unknown')
            difficulty = result.get('metadata', {}).get('difficulty', 'unknown')
            
            if 'combined_score' in result:
                score = result['combined_score']
            elif 'similarity' in result:
                score = result['similarity']
            elif 'score' in result:
                score = result['score']
            else:
                score = 0.0
                
            logger.info(f"      {count}. [{category}/{topic}/{difficulty}] Score: {score:.3f} | {result['content'][:80]}...")

def log_hybrid_weight_comparison(results_by_weight, query):
    """Log detailed comparison of different hybrid weights."""
    logger.info(f"🔍 HYBRID WEIGHT ANALYSIS for: '{query}'")
    logger.info("-" * 50)
    
    # Analyze each weight configuration
    for weight, results in results_by_weight.items():
        if results:
            unique_contents = set()
            unique_categories = set()
            score_range = []
            
            for result in results:
                content_preview = result['content'][:50]
                unique_contents.add(content_preview)
                unique_categories.add(str(result.get('metadata', {}).get('category', 'unknown')))
                score = result.get('combined_score', 0.0)
                score_range.append(score)
            
            top_score = max(score_range) if score_range else 0.0
            topic = results[0].get('metadata', {}).get('topic', 'unknown')
            category = results[0].get('metadata', {}).get('category', 'unknown')
            
            logger.info(f"   Weight {weight}:")
            logger.info(f"      Top Score: {top_score:.3f} | Category: {category} | Topic: {topic}")
            logger.info(f"      Unique Results: {len(unique_contents)}/{len(results)}")
            logger.info(f"      Categories: {len(unique_categories)}")
            logger.info(f"      Score Range: {min(score_range):.3f} - {max(score_range):.3f}")

def main():
    """Main function to demonstrate the RAG pipeline."""
    logger.info("🚀 RAG PIPELINE DEMONSTRATION STARTED")
    logger.info("=" * 80)
    
    # Initialize the RAG pipeline
    with RAGPipeline() as rag:
        initial_count = rag.get_document_count()
        logger.info(f"📊 Initial database state: {initial_count} documents")
        
        # Ingest sample documents
        logger.info("📥 Ingesting diverse sample documents...")
        sample_docs = create_sample_documents()
        doc_ids = rag.ingest_documents(sample_docs)
        final_count = rag.get_document_count()
        logger.info(f"✅ Successfully ingested {len(doc_ids)} documents")
        logger.info(f"📊 Database now contains: {final_count} documents")
        logger.info("")
        
        # Diverse test queries covering different topics and complexity levels
        test_queries = [
            # AI & Machine Learning
            "machine learning algorithms and neural networks",
            "deep learning for computer vision applications",
            "natural language processing and transformer models",
            "artificial intelligence ethics and bias",
            
            # Database Systems
            "PostgreSQL advanced features and performance",
            "NoSQL databases and MongoDB architecture",
            "vector databases for similarity search",
            "database indexing strategies and optimization",
            
            # RAG & Search
            "retrieval augmented generation implementation",
            "semantic search and embedding models",
            "hybrid search combining multiple techniques",
            "vector similarity and cosine distance",
            
            # Programming & Development
            "Python programming for data science",
            "Docker containers and microservices",
            "REST API design principles",
            "GraphQL vs REST comparison",
            
            # Data Science
            "data preprocessing and feature engineering",
            "pandas data manipulation techniques",
            "A/B testing and statistical analysis",
            "machine learning model evaluation",
            
            # Cloud & DevOps
            "AWS cloud services and architecture",
            "Kubernetes orchestration and scaling",
            "CI/CD pipeline automation",
            "infrastructure as code with Terraform",
            
            # Security
            "OAuth 2.0 authentication flow",
            "encryption algorithms and key management",
            "GDPR compliance and data protection"
        ]
        
        logger.info("🔍 COMPREHENSIVE SEARCH ANALYSIS PHASE")
        logger.info("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Test {i}/{len(test_queries)}: '{query}'")
            logger.info("-" * 50)
            
            # Hybrid search with analysis
            hybrid_results = rag.search(query, k=5, hybrid_weight=0.5)
            analyze_search_results(hybrid_results, "Hybrid Search", query)
            
            # Vector search with analysis
            vector_results = rag.vector_search(query, k=5)
            analyze_search_results(vector_results, "Vector Search", query)
            
            # Full-text search with analysis
            text_results = rag.full_text_search(query, k=5)
            analyze_search_results(text_results, "Full-Text Search", query)
            
            logger.info("")  # Empty line for readability
        
        # Detailed hybrid weight analysis with diverse queries
        logger.info("⚖️ DETAILED HYBRID WEIGHT ANALYSIS")
        logger.info("=" * 60)
        
        weight_test_queries = [
            "machine learning",
            "database systems", 
            "cloud computing",
            "data science"
        ]
        
        for test_query in weight_test_queries:
            logger.info(f"Testing weights for: '{test_query}'")
            results_by_weight = {}
            
            for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
                results = rag.search(test_query, k=5, hybrid_weight=weight)
                results_by_weight[weight] = results
            
            log_hybrid_weight_comparison(results_by_weight, test_query)
            logger.info("")
        
        # Document retrieval test
        logger.info("📄 DOCUMENT RETRIEVAL TEST")
        logger.info("-" * 40)
        
        if hybrid_results:
            node_id = hybrid_results[0]['id']
            logger.info(f"Retrieving document with Node ID: {node_id}")
            doc = rag.get_document_by_node_id(node_id)
            
            if doc:
                topic = doc.get('metadata', {}).get('topic', 'unknown')
                category = doc.get('metadata', {}).get('category', 'unknown')
                difficulty = doc.get('metadata', {}).get('difficulty', 'unknown')
                content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                
                logger.info(f"✅ Retrieved: [{category}/{topic}/{difficulty}]")
                logger.info(f"   Content: {content_preview}")
            else:
                logger.error(f"❌ Failed to retrieve document with Node ID: {node_id}")
        else:
            logger.warning("❌ No search results available for document retrieval test")
    
    logger.info("")
    logger.info("✅ RAG PIPELINE DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()

