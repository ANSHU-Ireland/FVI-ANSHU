#!/usr/bin/env python3
"""
Script to initialize the database schema using SQLAlchemy models.
"""
import sys
import os
import time

# Add the src directory to path
sys.path.append('/workspaces/FVI-ANSHU/src')

from database.connection import engine, SessionLocal
from models.database import Base
from sqlalchemy import text

def init_database():
    """Initialize the database schema."""
    print("Initializing database schema...")
    
    try:
        # Wait for database to be ready
        print("Waiting for database to be ready...")
        time.sleep(5)
        
        # Create all tables
        print("Creating tables...")
        Base.metadata.create_all(bind=engine)
        
        # Create indexes and extensions
        print("Creating indexes and extensions...")
        with engine.connect() as conn:
            # Create extensions
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"))
            
            # Create indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_data_sources_name ON data_sources(name);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_definitions_name ON metric_definitions(name);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_definitions_category ON metric_definitions(category);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_values_metric_id ON metric_values(metric_definition_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_values_entity ON metric_values(sub_industry, country);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_values_time ON metric_values(year, quarter, month);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_composite_scores_entity ON composite_scores(sub_industry, country);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_composite_scores_time ON composite_scores(year, horizon);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_runs_name ON model_runs(model_name);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_model_runs_deployed ON model_runs(is_deployed);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);"))
            
            # Create full-text search indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_definitions_description_fts ON metric_definitions USING gin(to_tsvector('english', description));"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_data_sources_description_fts ON data_sources USING gin(to_tsvector('english', description));"))
            
            # Create composite indexes for common queries
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_metric_values_entity_metric_time ON metric_values(sub_industry, metric_definition_id, year);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_composite_scores_entity_horizon_time ON composite_scores(sub_industry, horizon, year);"))
            
            conn.commit()
        
        print("Database initialization complete!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if init_database():
        print("Database ready for data population!")
    else:
        print("Database initialization failed!")
        sys.exit(1)
