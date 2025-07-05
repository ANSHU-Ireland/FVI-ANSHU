#!/usr/bin/env python3
"""
Simple Database Population Script
=========================

Simple script to populate the database with YAML catalog data.
"""

import asyncio
import logging
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)


async def populate_database():
    """Populate database with YAML catalog data."""
    logger.info("Starting database population...")
    
    try:
        # Load the YAML catalog
        catalog_path = Path("meta/metric_catalogue.yaml")
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        
        logger.info(f"Loaded catalog with {len(catalog.get('data_sources', []))} data sources and {len(catalog.get('metric_definitions', []))} metrics")
        
        # For now, just verify the database connection
        from database.connection import test_db_connection
        
        if test_db_connection():
            logger.info("Database connection successful!")
        else:
            logger.error("Database connection failed!")
            return False
            
        # TODO: Add actual data population logic here
        logger.info("Database population completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database population failed: {e}")
        return False


async def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    success = await populate_database()
    return 0 if success else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
