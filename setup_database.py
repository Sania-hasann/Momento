#!/usr/bin/env python3
"""
Database setup script for the RAG pipeline using LlamaIndex.
"""

import psycopg2
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_db_config():
    """Get database configuration from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "rag_database"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def test_connection(config):
    """Test database connection."""
    try:
        conn = psycopg2.connect(**config)
        conn.close()
        print("✅ Database connection successful!")
        return True
    except psycopg2.Error as e:
        print(f"❌ Database connection failed: {e}")
        return False


def create_database(config):
    """Create the database if it doesn't exist."""
    # Connect to default postgres database to create our database
    create_config = config.copy()
    create_config["database"] = "postgres"

    try:
        conn = psycopg2.connect(**create_config)
        conn.autocommit = True

        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (config["database"],)
            )
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(f"CREATE DATABASE {config['database']}")
                print(f"✅ Database '{config['database']}' created successfully!")
            else:
                print(f"ℹ️  Database '{config['database']}' already exists.")

        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"❌ Error creating database: {e}")
        return False


def setup_pgvector(config):
    """Set up pgvector extension."""
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True

        with conn.cursor() as cursor:
            # Check if pgvector extension is available
            cursor.execute(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
            )
            available = cursor.fetchone()

            if not available:
                print(
                    "❌ pgvector extension is not available. Please install it first."
                )
                print("   See README.md for installation instructions.")
                return False

            # Create extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ pgvector extension enabled successfully!")

        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"❌ Error setting up pgvector: {e}")
        return False


def verify_setup(config):
    """Verify the setup by checking extensions."""
    try:
        conn = psycopg2.connect(**config)

        with conn.cursor() as cursor:
            # Check pgvector extension
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cursor.fetchone():
                print("✅ pgvector extension is active")
            else:
                print("❌ pgvector extension is not active")
                return False

            # Check if vector type exists
            cursor.execute("SELECT 1 FROM pg_type WHERE typname = 'vector'")
            if cursor.fetchone():
                print("✅ vector type is available")
            else:
                print("❌ vector type is not available")
                return False

        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"❌ Error verifying setup: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 RAG Pipeline Database Setup (LlamaIndex)")
    print("=" * 60)

    # Get configuration
    config = get_db_config()
    print(f"📋 Configuration:")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Database: {config['database']}")
    print(f"   User: {config['user']}")
    print()

    # Test connection
    print("🔌 Testing database connection...")
    if not test_connection(config):
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure PostgreSQL is running")
        print("   2. Check your .env file configuration")
        print("   3. Verify database credentials")
        sys.exit(1)

    # Create database
    print("\n🗄️  Setting up database...")
    if not create_database(config):
        sys.exit(1)

    # Setup pgvector
    print("\n🔧 Setting up pgvector extension...")
    if not setup_pgvector(config):
        sys.exit(1)

    # Verify setup
    print("\n✅ Verifying setup...")
    if not verify_setup(config):
        print("❌ Setup verification failed")
        sys.exit(1)

    print("\n🎉 Database setup completed successfully!")
    print("\n📝 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run 'python example.py' to test the pipeline")
    print("   3. Check the README.md for usage examples")
    print("   4. Start ingesting your documents!")


if __name__ == "__main__":
    main()
