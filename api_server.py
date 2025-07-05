#!/usr/bin/env python3
"""
Simple API Server
================

Simple FastAPI server for testing the FVI system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global catalog path
CATALOG_PATH = Path("/workspaces/FVI-ANSHU/meta/metric_catalogue.yaml")

app = FastAPI(
    title="FVI Analytics API",
    description="Future Viability Index Analytics Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str


class CatalogSummary(BaseModel):
    total_data_sources: int
    total_metrics: int
    sheets_processed: int
    metrics_by_sheet: Dict[str, int]


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    from datetime import datetime
    return HealthResponse(
        status="healthy",
        message="FVI Analytics API is running",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        from database.connection import test_db_connection
        db_healthy = test_db_connection()
        
        from datetime import datetime
        return HealthResponse(
            status="healthy" if db_healthy else "unhealthy",
            message="Database connection successful" if db_healthy else "Database connection failed",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        from datetime import datetime
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {e}",
            timestamp=datetime.now().isoformat()
        )


@app.get("/catalog", response_model=CatalogSummary)
async def get_catalog_summary():
    """Get summary of the metric catalog."""
    try:
        import yaml
        catalog_path = Path(__file__).parent / "meta" / "metric_catalogue.yaml"
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        
        return CatalogSummary(
            total_data_sources=len(catalog.get("data_sources", [])),
            total_metrics=len(catalog.get("metric_definitions", [])),
            sheets_processed=len(catalog.get("sheet_mappings", {})),
            metrics_by_sheet=catalog.get("summary", {}).get("metrics_by_sheet", {})
        )
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load catalog: {e}")


@app.get("/data-sources")
async def get_data_sources():
    """Get list of data sources."""
    try:
        import yaml
        catalog_path = Path(__file__).parent / "meta" / "metric_catalogue.yaml"
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        
        return catalog.get("data_sources", [])
    except Exception as e:
        logger.error(f"Failed to load data sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load data sources: {e}")


@app.get("/metrics")
async def get_metrics():
    """Get list of metric definitions."""
    try:
        import yaml
        catalog_path = Path(__file__).parent / "meta" / "metric_catalogue.yaml"
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        
        return catalog.get("metric_definitions", [])
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load metrics: {e}")


@app.get("/metrics/{metric_key}")
async def get_metric(metric_key: str):
    """Get specific metric definition."""
    try:
        import yaml
        catalog_path = Path(__file__).parent / "meta" / "metric_catalogue.yaml"
        
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog = yaml.safe_load(f)
        
        for metric in catalog.get("metric_definitions", []):
            if metric.get("metric_key") == metric_key:
                return metric
        
        raise HTTPException(status_code=404, detail=f"Metric {metric_key} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load metric: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
