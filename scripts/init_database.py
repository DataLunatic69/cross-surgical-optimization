"""
Initialize database with tables and schemas.
This script can be run independently to set up the database.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.db.base import Base, engine
from app.core.config import settings
from app.utils.logger import app_logger

# Import all models to ensure they're registered
from app.models import *  # noqa


def create_schemas():
    """Create database schemas if they don't exist."""
    schemas = ['auth', 'hospital', 'training', 'prediction', 'analytics']
    
    with engine.begin() as conn:  # Use begin() for automatic transaction handling
        for schema in schemas:
            try:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                app_logger.info(f"✓ Schema '{schema}' created or already exists")
            except Exception as e:
                app_logger.error(f"✗ Error creating schema '{schema}': {e}")
                raise


def create_tables():
    """Create all tables from SQLAlchemy models."""
    try:
        # This will create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        app_logger.info("✓ All tables created successfully")
        return True
    except Exception as e:
        app_logger.error(f"✗ Error creating tables: {e}")
        return False


def verify_tables():
    """Verify that all tables were created."""
    with engine.connect() as conn:
        # Check each schema for tables
        schemas = ['auth', 'hospital', 'training', 'prediction']
        
        for schema in schemas:
            result = conn.execute(text(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            
            if tables:
                app_logger.info(f"✓ Schema '{schema}' contains tables: {', '.join(tables)}")
            else:
                app_logger.warning(f"⚠ Schema '{schema}' has no tables")


def create_extensions():
    """Create PostgreSQL extensions."""
    extensions = ['uuid-ossp', 'pgcrypto']
    
    with engine.begin() as conn:  # Use begin() for automatic transaction handling
        for ext in extensions:
            try:
                conn.execute(text(f'CREATE EXTENSION IF NOT EXISTS "{ext}"'))
                app_logger.info(f"✓ Extension '{ext}' created or already exists")
            except Exception as e:
                app_logger.warning(f"⚠ Could not create extension '{ext}': {e}")
                # Don't raise, extensions might already exist


def verify_database_connection():
    """Verify we can connect to the database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            app_logger.info("✓ Database connection verified")
            return True
    except Exception as e:
        app_logger.error(f"✗ Cannot connect to database: {e}")
        return False


def main():
    """Main initialization function."""
    app_logger.info("=" * 50)
    app_logger.info("Database Initialization Starting")
    app_logger.info("=" * 50)
    app_logger.info(f"Database: {settings.DATABASE_URL.split('@')[1]}")
    
    # Step 0: Verify connection
    app_logger.info("\n0. Verifying database connection...")
    if not verify_database_connection():
        app_logger.error("Cannot connect to database. Please check your connection settings.")
        return False
    
    # Step 1: Create extensions
    app_logger.info("\n1. Creating PostgreSQL extensions...")
    create_extensions()
    
    # Step 2: Create schemas
    app_logger.info("\n2. Creating database schemas...")
    create_schemas()
    
    # Step 3: Create tables
    app_logger.info("\n3. Creating tables from models...")
    success = create_tables()
    
    # Step 4: Verify creation
    if success:
        app_logger.info("\n4. Verifying table creation...")
        verify_tables()
    
    app_logger.info("\n" + "=" * 50)
    if success:
        app_logger.info("✓ Database initialization completed successfully!")
        app_logger.info("\nYou can now run Alembic migrations to track schema changes:")
        app_logger.info("  alembic stamp head  # Mark current schema as up-to-date")
        app_logger.info("  alembic revision --autogenerate -m 'Your message'  # Create new migrations")
    else:
        app_logger.error("✗ Database initialization failed. Check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)