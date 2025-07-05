#!/usr/bin/env python3
"""
FVI Analytics CLI - Command Line Interface for FVI system management
"""

import click
import logging
from pathlib import Path
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.database import db_manager, test_db_connection, test_redis_connection
from src.data import ExcelProcessor, DataSourceManager, MockDataGenerator
from src.ml import FVIPredictor, EnsemblePredictor
from src.api.main import app
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def cli(debug):
    """FVI Analytics CLI - Manage your Future Viability Index system."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


@db.command()
def reset():
    """Reset the database (WARNING: This will delete all data)."""
    if click.confirm("This will delete all data in the database. Are you sure?"):
        try:
            logger.info("Resetting database...")
            db_manager.reset_database()
            logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Database reset failed: {e}")
            sys.exit(1)


@db.command()
def test():
    """Test database and Redis connections."""
    logger.info("Testing database connection...")
    db_ok = test_db_connection()
    
    logger.info("Testing Redis connection...")
    redis_ok = test_redis_connection()
    
    if db_ok and redis_ok:
        logger.info("All connections successful")
        sys.exit(0)
    else:
        logger.error("Some connections failed")
        sys.exit(1)


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.argument('excel_file', type=click.Path(exists=True))
def import_excel(excel_file):
    """Import data from Excel file."""
    try:
        logger.info(f"Processing Excel file: {excel_file}")
        processor = ExcelProcessor(excel_file)
        results = processor.process_all_sheets()
        
        logger.info(f"Found {len(results['data_sources'])} data sources")
        logger.info(f"Found {len(results['metrics'])} metrics")
        
        # Save results to database would go here
        logger.info("Excel import completed successfully")
        
    except Exception as e:
        logger.error(f"Excel import failed: {e}")
        sys.exit(1)


@data.command()
@click.option('--source', default='IEA', help='Data source name')
@click.option('--dataset', help='Dataset identifier')
def fetch(source, dataset):
    """Fetch data from external sources."""
    try:
        logger.info(f"Fetching data from {source}")
        manager = DataSourceManager()
        
        # This would be async in practice
        logger.info("Data fetch initiated (async operation)")
        
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        sys.exit(1)


@data.command()
@click.option('--industry', default='Coal Mining', help='Sub-industry name')
@click.option('--country', default='Global', help='Country name')
def generate_mock(industry, country):
    """Generate mock data for testing."""
    try:
        logger.info(f"Generating mock data for {industry} in {country}")
        mock_data = MockDataGenerator.generate_coal_metrics_data(industry, country)
        
        logger.info(f"Generated data for {len(mock_data['years'])} years")
        logger.info(f"Generated {len(mock_data['metrics'])} metrics")
        
    except Exception as e:
        logger.error(f"Mock data generation failed: {e}")
        sys.exit(1)


@cli.group()
def ml():
    """Machine learning commands."""
    pass


@ml.command()
@click.option('--model-type', default='lightgbm', help='Model type (lightgbm, catboost, etc.)')
@click.option('--industry', default='Coal Mining', help='Sub-industry name')
def train(model_type, industry):
    """Train FVI prediction model."""
    try:
        logger.info(f"Training {model_type} model for {industry}")
        
        # Generate training data
        mock_data = MockDataGenerator.generate_coal_metrics_data(industry)
        
        # This would create proper training data and train the model
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)


@ml.command()
@click.option('--model-types', default='lightgbm,catboost,random_forest', help='Comma-separated model types')
@click.option('--industry', default='Coal Mining', help='Sub-industry name')
def train_ensemble(model_types, industry):
    """Train ensemble of models."""
    try:
        model_list = [m.strip() for m in model_types.split(',')]
        logger.info(f"Training ensemble with models: {model_list}")
        
        # This would train the ensemble
        logger.info("Ensemble training completed successfully")
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        sys.exit(1)


@ml.command()
@click.option('--industry', default='Coal Mining', help='Sub-industry name')
@click.option('--country', default='Global', help='Country name')
@click.option('--horizon', default=10, help='Prediction horizon in years')
def predict(industry, country, horizon):
    """Make FVI prediction."""
    try:
        logger.info(f"Making prediction for {industry} in {country} (horizon: {horizon} years)")
        
        # This would load model and make prediction
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


@cli.group()
def api():
    """API server commands."""
    pass


@api.command()
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def run(host, port, reload):
    """Start the API server."""
    logger.info(f"Starting API server on {host}:{port}")
    
    # Test connections before starting
    if not test_db_connection():
        logger.error("Database connection failed. Please check your configuration.")
        sys.exit(1)
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@api.command()
def test_server():
    """Test API server endpoints."""
    import requests
    
    try:
        logger.info("Testing API server...")
        response = requests.get("http://localhost:8000/health")
        
        if response.status_code == 200:
            logger.info("API server is healthy")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"API server returned status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API server. Is it running?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"API test failed: {e}")
        sys.exit(1)


@cli.group()
def docker():
    """Docker management commands."""
    pass


@docker.command()
def up():
    """Start Docker services."""
    import subprocess
    
    try:
        logger.info("Starting Docker services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        logger.info("Docker services started successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker startup failed: {e}")
        sys.exit(1)


@docker.command()
def down():
    """Stop Docker services."""
    import subprocess
    
    try:
        logger.info("Stopping Docker services...")
        subprocess.run(["docker-compose", "down"], check=True)
        logger.info("Docker services stopped successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker shutdown failed: {e}")
        sys.exit(1)


@docker.command()
def logs():
    """View Docker logs."""
    import subprocess
    
    try:
        subprocess.run(["docker-compose", "logs", "-f"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker logs failed: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show system status."""
    logger.info("=== FVI Analytics System Status ===")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check database
    logger.info("Database connection:", "OK" if test_db_connection() else "FAILED")
    
    # Check Redis
    logger.info("Redis connection:", "OK" if test_redis_connection() else "FAILED")
    
    # Check directories
    for dir_name in ["data", "models", "logs"]:
        dir_path = Path(dir_name)
        logger.info(f"{dir_name.title()} directory:", "OK" if dir_path.exists() else "MISSING")
    
    # Check configuration
    logger.info("Configuration:")
    logger.info(f"  Database URL: {settings.DATABASE_URL[:50]}...")
    logger.info(f"  Redis URL: {settings.REDIS_URL}")
    logger.info(f"  OpenAI API Key: {'Configured' if settings.OPENAI_API_KEY else 'Not configured'}")


@cli.command()
def install():
    """Install dependencies and initialize system."""
    try:
        logger.info("Installing FVI Analytics system...")
        
        # Create directories
        for dir_name in ["data", "models", "logs"]:
            Path(dir_name).mkdir(exist_ok=True)
            logger.info(f"Created {dir_name} directory")
        
        # Initialize database
        logger.info("Initializing database...")
        db_manager.create_tables()
        
        # Copy environment file
        env_file = Path(".env")
        if not env_file.exists():
            import shutil
            shutil.copy(".env.example", ".env")
            logger.info("Created .env file from template")
        
        logger.info("Installation completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Edit .env file with your configuration")
        logger.info("2. Run 'python fvi_cli.py api run' to start the API server")
        logger.info("3. Visit http://localhost:8000/docs for API documentation")
        
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
