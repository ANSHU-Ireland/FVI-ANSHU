#!/usr/bin/env python3
"""
Database Population Script
=========================

Populates the FVI database with data from the YAML catalog
and runs the ETL pipeline to process the Excel data.

Usage:
    python scripts/populate_database.py
"""

import asyncio
import logging
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_settings
from database.connection import DatabaseManager
from database.crud import FVICrudOperations
from models.schemas import (
    MetricDefinitionCreate, DataSourceCreate, CompanyCreate, 
    MetricDataCreate, HorizonWeightCreate
)
from data.processor import FVIDataProcessor
from data.sources import DataSourceManager

logger = logging.getLogger(__name__)


class DatabasePopulator:
    """Populate database with YAML catalog data."""
    
    def __init__(self, catalog_path: str = "meta/metric_catalogue.yaml"):
        self.catalog_path = Path(catalog_path)
        self.settings = get_settings()
        self.db_manager = DatabaseManager(self.settings.DATABASE_URL)
        self.crud = FVICrudOperations(self.db_manager)
        self.data_processor = FVIDataProcessor()
        self.data_source_manager = DataSourceManager()
        
    async def load_catalog(self) -> dict:
        """Load the YAML catalog."""
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    async def populate_data_sources(self, catalog: dict) -> None:
        """Populate data sources from catalog."""
        logger.info("Populating data sources...")
        
        for ds_data in catalog.get("data_sources", []):
            # Create data source
            data_source = DataSourceCreate(
                name=ds_data["name"],
                source_type="external",
                connection_string=ds_data.get("source_link", ""),
                description=ds_data.get("description", ""),
                metadata=ds_data.get("metadata", {})
            )
            
            try:
                await self.crud.create_data_source(data_source)
                logger.info(f"Created data source: {data_source.name}")
            except Exception as e:
                logger.warning(f"Failed to create data source {data_source.name}: {e}")
                
    async def populate_metric_definitions(self, catalog: dict) -> None:
        """Populate metric definitions from catalog."""
        logger.info("Populating metric definitions...")
        
        for metric_data in catalog.get("metric_definitions", []):
            # Create metric definition
            metric_def = MetricDefinitionCreate(
                metric_key=metric_data["metric_key"],
                feature_column=metric_data["feature_column"],
                sheet_name=metric_data["sheet_info"]["sheet_name"],
                sheet_number=metric_data["sheet_info"]["sheet_number"],
                thematic_focus=metric_data["sheet_info"]["thematic_focus"],
                title=metric_data["basic_info"]["title"],
                slug=metric_data["basic_info"]["slug"],
                details=metric_data["basic_info"].get("details"),
                formula=metric_data["basic_info"].get("formula"),
                data_fields_required=metric_data["basic_info"].get("data_fields_required"),
                weighting_in_metric=metric_data["basic_info"].get("weighting_in_metric", 0.0),
                data_source=metric_data["basic_info"].get("data_source"),
                structured_availability=metric_data["data_quality"]["structured_availability"],
                unstructured_availability=metric_data["data_quality"]["unstructured_availability"],
                country_level_availability=metric_data["data_quality"]["country_level_availability"],
                sub_industry_availability=metric_data["data_quality"]["sub_industry_availability"],
                volume_of_data=metric_data["data_quality"]["volume_of_data"],
                alternative_proxy_feasibility=metric_data["data_quality"]["alternative_proxy_feasibility"],
                genai_ml_fillability=metric_data["data_quality"]["genai_ml_fillability"],
                industry_level_variability=metric_data["data_quality"]["industry_level_variability"],
                longitudinal_availability=metric_data["data_quality"]["longitudinal_availability"],
                data_verification_bias_risk=metric_data["data_quality"]["data_verification_bias_risk"],
                interdependence_with_other_metrics=metric_data["data_quality"]["interdependence_with_other_metrics"],
                overlap_with_other_metrics=metric_data["assessment"].get("overlap_with_other_metrics"),
                scoring_methodology_notes=metric_data["assessment"].get("scoring_methodology_notes"),
                readiness_status=metric_data["assessment"].get("readiness_status", "Draft"),
                linked_sheet_or_tab=metric_data["assessment"].get("linked_sheet_or_tab")
            )
            
            try:
                await self.crud.create_metric_definition(metric_def)
                logger.info(f"Created metric definition: {metric_def.metric_key}")
            except Exception as e:
                logger.warning(f"Failed to create metric definition {metric_def.metric_key}: {e}")
                
    async def populate_sample_companies(self) -> None:
        """Populate sample companies for testing."""
        logger.info("Populating sample companies...")
        
        sample_companies = [
            {
                "name": "Example Coal Corp",
                "ticker": "COAL",
                "industry": "Coal Mining",
                "sub_industry": "Thermal Coal",
                "country": "United States",
                "region": "North America",
                "market_cap": 1000000000.0,
                "is_active": True,
                "metadata": {"sample": True}
            },
            {
                "name": "Global Energy Ltd",
                "ticker": "GENL",
                "industry": "Coal Mining",
                "sub_industry": "Metallurgical Coal",
                "country": "Australia",
                "region": "Asia-Pacific",
                "market_cap": 2500000000.0,
                "is_active": True,
                "metadata": {"sample": True}
            },
            {
                "name": "Mining Resources Inc",
                "ticker": "MRIN",
                "industry": "Coal Mining",
                "sub_industry": "Mixed Coal",
                "country": "Poland",
                "region": "Europe",
                "market_cap": 500000000.0,
                "is_active": True,
                "metadata": {"sample": True}
            }
        ]
        
        for company_data in sample_companies:
            company = CompanyCreate(**company_data)
            try:
                await self.crud.create_company(company)
                logger.info(f"Created company: {company.name}")
            except Exception as e:
                logger.warning(f"Failed to create company {company.name}: {e}")
                
    async def populate_sample_metric_data(self) -> None:
        """Populate sample metric data for testing."""
        logger.info("Populating sample metric data...")
        
        # Get companies and metrics
        companies = await self.crud.get_companies()
        metrics = await self.crud.get_metric_definitions()
        
        if not companies or not metrics:
            logger.warning("No companies or metrics found. Skipping sample data generation.")
            return
            
        import random
        import datetime
        
        # Generate sample data for first 10 metrics and all companies
        for company in companies[:3]:  # Limit to first 3 companies
            for metric in metrics[:10]:  # Limit to first 10 metrics
                # Generate random sample values
                value = random.uniform(0.0, 100.0)
                normalized_value = value / 100.0
                
                metric_data = MetricDataCreate(
                    company_id=company.id,
                    metric_definition_id=metric.id,
                    period_start=datetime.date(2023, 1, 1),
                    period_end=datetime.date(2023, 12, 31),
                    value=value,
                    normalized_value=normalized_value,
                    data_source_id=None,  # Will be set later
                    metadata={"sample": True, "generated": True}
                )
                
                try:
                    await self.crud.create_metric_data(metric_data)
                    logger.debug(f"Created metric data for {company.name} - {metric.metric_key}")
                except Exception as e:
                    logger.warning(f"Failed to create metric data: {e}")
                    
        logger.info("Sample metric data populated")
        
    async def populate_sample_weights(self) -> None:
        """Populate sample horizon weights."""
        logger.info("Populating sample horizon weights...")
        
        # Get companies and metrics
        companies = await self.crud.get_companies()
        metrics = await self.crud.get_metric_definitions()
        
        if not companies or not metrics:
            logger.warning("No companies or metrics found. Skipping weight generation.")
            return
            
        import random
        
        horizons = ["H5", "H10", "H20"]
        
        for company in companies[:3]:  # Limit to first 3 companies
            for metric in metrics[:10]:  # Limit to first 10 metrics
                for horizon in horizons:
                    # Generate random weights that sum to reasonable values
                    weight = random.uniform(0.1, 1.0)
                    
                    horizon_weight = HorizonWeightCreate(
                        company_id=company.id,
                        metric_definition_id=metric.id,
                        horizon=horizon,
                        weight=weight,
                        methodology="Random Sample",
                        metadata={"sample": True}
                    )
                    
                    try:
                        await self.crud.create_horizon_weight(horizon_weight)
                        logger.debug(f"Created weight for {company.name} - {metric.metric_key} - {horizon}")
                    except Exception as e:
                        logger.warning(f"Failed to create horizon weight: {e}")
                        
        logger.info("Sample horizon weights populated")
        
    async def run_etl_pipeline(self) -> None:
        """Run the ETL pipeline to process Excel data."""
        logger.info("Running ETL pipeline...")
        
        try:
            # Process Excel file
            excel_path = Path("FVI Scoring Metrics_Coal.xlsx")
            if excel_path.exists():
                await self.data_processor.process_excel_file(str(excel_path))
                logger.info("Excel file processed successfully")
            else:
                logger.warning("Excel file not found, skipping ETL pipeline")
                
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            
    async def populate_all(self) -> None:
        """Populate all database tables."""
        logger.info("Starting database population...")
        
        try:
            # Load catalog
            catalog = await self.load_catalog()
            logger.info(f"Loaded catalog with {len(catalog.get('data_sources', []))} data sources and {len(catalog.get('metric_definitions', []))} metrics")
            
            # Populate tables
            await self.populate_data_sources(catalog)
            await self.populate_metric_definitions(catalog)
            await self.populate_sample_companies()
            await self.populate_sample_metric_data()
            await self.populate_sample_weights()
            
            # Run ETL pipeline
            await self.run_etl_pipeline()
            
            logger.info("Database population completed successfully!")
            
        except Exception as e:
            logger.error(f"Database population failed: {e}")
            raise
            
        finally:
            await self.db_manager.close()


async def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        populator = DatabasePopulator()
        await populator.populate_all()
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
