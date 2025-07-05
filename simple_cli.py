#!/usr/bin/env python3
"""
FVI CLI - Simple Version
========================

Command line interface for FVI system operations.
"""

import sys
import os
import argparse
import logging
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description="FVI Analytics CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Database commands
    db_parser = subparsers.add_parser('db', help='Database operations')
    db_subparsers = db_parser.add_subparsers(dest='db_command')
    
    db_subparsers.add_parser('test', help='Test database connection')
    db_subparsers.add_parser('init', help='Initialize database')
    db_subparsers.add_parser('status', help='Check database status')
    
    # Data commands
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_subparsers = data_parser.add_subparsers(dest='data_command')
    
    data_subparsers.add_parser('convert', help='Convert Excel to YAML')
    data_subparsers.add_parser('validate', help='Validate data quality')
    data_subparsers.add_parser('summary', help='Show data summary')
    
    # API commands
    api_parser = subparsers.add_parser('api', help='API operations')
    api_subparsers = api_parser.add_subparsers(dest='api_command')
    
    api_subparsers.add_parser('start', help='Start API server')
    api_subparsers.add_parser('test', help='Test API endpoints')
    api_subparsers.add_parser('docs', help='Show API documentation URL')
    
    # System commands
    system_parser = subparsers.add_parser('system', help='System operations')
    system_subparsers = system_parser.add_subparsers(dest='system_command')
    
    system_subparsers.add_parser('test', help='Run system tests')
    system_subparsers.add_parser('status', help='Check system status')
    system_subparsers.add_parser('info', help='Show system information')
    
    return parser


def db_operations(args):
    """Handle database operations."""
    if args.db_command == 'test':
        logger.info("Testing database connection...")
        try:
            from database.connection import test_db_connection
            if test_db_connection():
                logger.info("✓ Database connection successful")
            else:
                logger.error("✗ Database connection failed")
        except Exception as e:
            logger.error(f"✗ Database test failed: {e}")
            
    elif args.db_command == 'status':
        logger.info("Checking database status...")
        try:
            from database.connection import test_db_connection
            if test_db_connection():
                logger.info("✓ Database is running")
            else:
                logger.error("✗ Database is not accessible")
        except Exception as e:
            logger.error(f"✗ Database status check failed: {e}")
            
    elif args.db_command == 'init':
        logger.info("Database initialization would be performed here")
        logger.info("(Not implemented in simple version)")
        
    else:
        logger.error("Unknown database command")


def data_operations(args):
    """Handle data operations."""
    if args.data_command == 'convert':
        logger.info("Converting Excel to YAML...")
        try:
            import subprocess
            result = subprocess.run([
                'python', 'scripts/excel2yaml.py', 
                'FVI Scoring Metrics_Coal.xlsx', 
                '--out', 'meta/metric_catalogue.yaml'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Excel to YAML conversion successful")
            else:
                logger.error(f"✗ Conversion failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"✗ Conversion failed: {e}")
            
    elif args.data_command == 'validate':
        logger.info("Validating data quality...")
        try:
            import subprocess
            result = subprocess.run([
                'python', 'test_system.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Data validation successful")
            else:
                logger.error(f"✗ Data validation failed")
                
        except Exception as e:
            logger.error(f"✗ Data validation failed: {e}")
            
    elif args.data_command == 'summary':
        logger.info("Showing data summary...")
        try:
            import yaml
            catalog_path = Path("meta/metric_catalogue.yaml")
            
            if catalog_path.exists():
                with open(catalog_path, 'r') as f:
                    catalog = yaml.safe_load(f)
                    
                logger.info(f"Data Sources: {len(catalog.get('data_sources', []))}")
                logger.info(f"Metrics: {len(catalog.get('metric_definitions', []))}")
                logger.info(f"Sheets: {len(catalog.get('sheet_mappings', {}))}")
                
                # Show metrics by sheet
                summary = catalog.get('summary', {})
                if 'metrics_by_sheet' in summary:
                    logger.info("Metrics by sheet:")
                    for sheet, count in summary['metrics_by_sheet'].items():
                        logger.info(f"  {sheet}: {count}")
                        
            else:
                logger.error("✗ Catalog file not found")
                
        except Exception as e:
            logger.error(f"✗ Failed to show summary: {e}")
            
    else:
        logger.error("Unknown data command")


def api_operations(args):
    """Handle API operations."""
    if args.api_command == 'start':
        logger.info("Starting API server...")
        try:
            import subprocess
            subprocess.run(['python', 'api_server.py'])
        except KeyboardInterrupt:
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"✗ Failed to start API server: {e}")
            
    elif args.api_command == 'test':
        logger.info("Testing API endpoints...")
        try:
            import requests
            
            # Test if API is running
            response = requests.get('http://localhost:8000/', timeout=5)
            if response.status_code == 200:
                logger.info("✓ API server is running")
                
                # Test catalog endpoint
                response = requests.get('http://localhost:8000/catalog', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✓ Catalog endpoint working: {data['total_metrics']} metrics")
                else:
                    logger.error("✗ Catalog endpoint failed")
                    
            else:
                logger.error("✗ API server not responding")
                
        except Exception as e:
            logger.error(f"✗ API test failed: {e}")
            
    elif args.api_command == 'docs':
        logger.info("API Documentation:")
        logger.info("  Swagger UI: http://localhost:8000/docs")
        logger.info("  ReDoc: http://localhost:8000/redoc")
        logger.info("  OpenAPI JSON: http://localhost:8000/openapi.json")
        
    else:
        logger.error("Unknown API command")


def system_operations(args):
    """Handle system operations."""
    if args.system_command == 'test':
        logger.info("Running system tests...")
        try:
            import subprocess
            result = subprocess.run(['python', 'test_system.py'], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ All system tests passed")
            else:
                logger.error("✗ Some system tests failed")
                
        except Exception as e:
            logger.error(f"✗ System test failed: {e}")
            
    elif args.system_command == 'status':
        logger.info("Checking system status...")
        
        # Check database
        try:
            from database.connection import test_db_connection
            db_ok = test_db_connection()
            logger.info(f"Database: {'✓' if db_ok else '✗'}")
        except:
            logger.info("Database: ✗")
            
        # Check API
        try:
            import requests
            response = requests.get('http://localhost:8000/', timeout=2)
            api_ok = response.status_code == 200
            logger.info(f"API Server: {'✓' if api_ok else '✗'}")
        except:
            logger.info("API Server: ✗")
            
        # Check catalog
        catalog_ok = Path("meta/metric_catalogue.yaml").exists()
        logger.info(f"Catalog: {'✓' if catalog_ok else '✗'}")
        
    elif args.system_command == 'info':
        logger.info("FVI Analytics System Information")
        logger.info("================================")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"System path: {sys.path[0]}")
        
        # Show key files
        key_files = [
            "meta/metric_catalogue.yaml",
            "FVI Scoring Metrics_Coal.xlsx",
            "requirements.txt",
            "docker-compose.yml"
        ]
        
        logger.info("\nKey files:")
        for file in key_files:
            exists = Path(file).exists()
            logger.info(f"  {file}: {'✓' if exists else '✗'}")
            
    else:
        logger.error("Unknown system command")


def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    try:
        if args.command == 'db':
            db_operations(args)
        elif args.command == 'data':
            data_operations(args)
        elif args.command == 'api':
            api_operations(args)
        elif args.command == 'system':
            system_operations(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
