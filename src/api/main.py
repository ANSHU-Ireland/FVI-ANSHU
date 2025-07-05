from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn

from database import get_db, init_db, test_db_connection
from models import (
    DataSourceCreate, DataSourceResponse,
    MetricDefinitionCreate, MetricDefinitionResponse,
    MetricValueCreate, MetricValueResponse,
    CompositeScoreCreate, CompositeScoreResponse,
    ModelRunResponse,
    FVIAnalysisRequest, FVIAnalysisResponse,
    ScenarioAnalysisRequest, ScenarioAnalysisResponse,
    ChatSessionCreate, ChatSessionResponse,
    ChatMessageCreate, ChatMessageResponse
)
from ..database import (
    DataSourceCRUD, MetricDefinitionCRUD, MetricValueCRUD,
    CompositeScoreCRUD, ModelRunCRUD, ChatCRUD
)
from ..config import settings
from .routes import data_routes, ml_routes, chat_routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FVI Analytics API",
    description="Future Viability Index (FVI) API for industry analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Include routers
app.include_router(data_routes.router, prefix="/api/v1/data", tags=["data"])
app.include_router(ml_routes.router, prefix="/api/v1/ml", tags=["machine-learning"])
app.include_router(chat_routes.router, prefix="/api/v1/chat", tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting FVI Analytics API")
    
    # Initialize database
    try:
        init_db()
        if test_db_connection():
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    logger.info("FVI Analytics API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FVI Analytics API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FVI Analytics API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": test_db_connection(),
            "api": True
        }
    }
    
    # Check if all services are healthy
    all_healthy = all(health_status["services"].values())
    health_status["status"] = "healthy" if all_healthy else "unhealthy"
    
    return health_status


@app.get("/info")
async def info():
    """API information endpoint."""
    return {
        "name": "FVI Analytics API",
        "version": "1.0.0",
        "description": "Future Viability Index (FVI) API for industry analysis",
        "features": [
            "Data source management",
            "Metric definition and tracking",
            "FVI score calculation",
            "Machine learning predictions",
            "Scenario analysis",
            "Chat-based analysis",
            "Explainable AI (SHAP)"
        ],
        "endpoints": {
            "data": "/api/v1/data",
            "ml": "/api/v1/ml",
            "chat": "/api/v1/chat"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


# Data Source endpoints
@app.post("/api/v1/data-sources", response_model=DataSourceResponse)
async def create_data_source(
    data_source: DataSourceCreate,
    db: Session = Depends(get_db)
):
    """Create a new data source."""
    try:
        db_data_source = DataSourceCRUD.create(db, data_source)
        return db_data_source
    except Exception as e:
        logger.error(f"Error creating data source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data-sources", response_model=List[DataSourceResponse])
async def get_data_sources(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all data sources."""
    try:
        data_sources = DataSourceCRUD.get_multi(db, skip=skip, limit=limit)
        return data_sources
    except Exception as e:
        logger.error(f"Error retrieving data sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data-sources/{data_source_id}", response_model=DataSourceResponse)
async def get_data_source(
    data_source_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific data source."""
    try:
        data_source = DataSourceCRUD.get(db, data_source_id)
        if not data_source:
            raise HTTPException(status_code=404, detail="Data source not found")
        return data_source
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving data source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metric Definition endpoints
@app.post("/api/v1/metrics", response_model=MetricDefinitionResponse)
async def create_metric_definition(
    metric: MetricDefinitionCreate,
    db: Session = Depends(get_db)
):
    """Create a new metric definition."""
    try:
        db_metric = MetricDefinitionCRUD.create(db, metric)
        return db_metric
    except Exception as e:
        logger.error(f"Error creating metric definition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics", response_model=List[MetricDefinitionResponse])
async def get_metric_definitions(
    skip: int = 0,
    limit: int = 100,
    sheet_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all metric definitions."""
    try:
        if sheet_name:
            metrics = MetricDefinitionCRUD.get_by_sheet(db, sheet_name)
        else:
            metrics = MetricDefinitionCRUD.get_multi(db, skip=skip, limit=limit)
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metric definitions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/{metric_id}", response_model=MetricDefinitionResponse)
async def get_metric_definition(
    metric_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific metric definition."""
    try:
        metric = MetricDefinitionCRUD.get(db, metric_id)
        if not metric:
            raise HTTPException(status_code=404, detail="Metric definition not found")
        return metric
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metric definition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metric Value endpoints
@app.post("/api/v1/metric-values", response_model=MetricValueResponse)
async def create_metric_value(
    metric_value: MetricValueCreate,
    db: Session = Depends(get_db)
):
    """Create a new metric value."""
    try:
        db_metric_value = MetricValueCRUD.create(db, metric_value)
        return db_metric_value
    except Exception as e:
        logger.error(f"Error creating metric value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metric-values", response_model=List[MetricValueResponse])
async def get_metric_values(
    skip: int = 0,
    limit: int = 100,
    metric_id: Optional[int] = None,
    sub_industry: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get metric values with optional filters."""
    try:
        if metric_id and sub_industry:
            metric_values = MetricValueCRUD.get_by_metric_and_entity(
                db, metric_id, sub_industry, country, year
            )
        else:
            metric_values = MetricValueCRUD.get_multi(db, skip=skip, limit=limit)
        return metric_values
    except Exception as e:
        logger.error(f"Error retrieving metric values: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Composite Score endpoints
@app.post("/api/v1/composite-scores", response_model=CompositeScoreResponse)
async def create_composite_score(
    composite_score: CompositeScoreCreate,
    db: Session = Depends(get_db)
):
    """Create a new composite score."""
    try:
        db_composite_score = CompositeScoreCRUD.create(db, composite_score)
        return db_composite_score
    except Exception as e:
        logger.error(f"Error creating composite score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/composite-scores", response_model=List[CompositeScoreResponse])
async def get_composite_scores(
    skip: int = 0,
    limit: int = 100,
    sub_industry: Optional[str] = None,
    country: Optional[str] = None,
    horizon: Optional[int] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get composite scores with optional filters."""
    try:
        if sub_industry and horizon:
            composite_scores = CompositeScoreCRUD.get_by_entity_and_horizon(
                db, sub_industry, horizon, country, year
            )
        elif sub_industry:
            composite_scores = CompositeScoreCRUD.get_latest_by_entity(
                db, sub_industry, country
            )
        else:
            composite_scores = CompositeScoreCRUD.get_multi(db, skip=skip, limit=limit)
        return composite_scores
    except Exception as e:
        logger.error(f"Error retrieving composite scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Run endpoints
@app.get("/api/v1/model-runs", response_model=List[ModelRunResponse])
async def get_model_runs(
    skip: int = 0,
    limit: int = 100,
    model_name: Optional[str] = None,
    is_deployed: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get model runs with optional filters."""
    try:
        if is_deployed is not None:
            model_runs = ModelRunCRUD.get_deployed_models(db) if is_deployed else []
        elif model_name:
            model_run = ModelRunCRUD.get_latest_by_model_name(db, model_name)
            model_runs = [model_run] if model_run else []
        else:
            model_runs = ModelRunCRUD.get_multi(db, skip=skip, limit=limit)
        return model_runs
    except Exception as e:
        logger.error(f"Error retrieving model runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
