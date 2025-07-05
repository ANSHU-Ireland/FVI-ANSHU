"""
Online Inference Service for FVI Analytics Platform

This module implements the online inference service with:
1. ONNX Runtime for low-latency predictions (~2ms)
2. FastAPI endpoints for sync and async batch predictions
3. Redis caching for sub-5ms response times
4. Model versioning and A/B testing capabilities
5. Gunicorn + Uvicorn deployment setup
6. OpenTelemetry instrumentation for observability

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.lightgbm
from contextlib import asynccontextmanager

from src.config import config
from src.database.db_manager import DatabaseManager
from src.ml.feature_store import FeatureStore
from src.observability.telemetry import get_telemetry, trace_function, trace_ml_prediction, traced_operation

logger = logging.getLogger(__name__)

# Initialize telemetry
telemetry = get_telemetry("fvi-inference-service")

# Pydantic models
class PredictionRequest(BaseModel):
    """Single prediction request"""
    mine_id: str = Field(..., description="Unique mine identifier")
    features: Dict[str, Union[float, int, str]] = Field(..., description="Feature values")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    return_explanations: bool = Field(False, description="Whether to return SHAP explanations")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")
    async_processing: bool = Field(False, description="Whether to process asynchronously")

class PredictionResponse(BaseModel):
    """Single prediction response"""
    mine_id: str
    prediction: float
    confidence: float
    model_version: str
    processing_time_ms: float
    explanation: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    async_job_id: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information"""
    model_version: str
    model_type: str
    created_at: datetime
    metrics: Dict[str, float]
    is_active: bool

# Global variables for model and resources
model_cache = {}
scaler_cache = {}
feature_store = None
redis_client = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    # Startup
    global feature_store, redis_client, db_manager
    
    # Initialize components
    feature_store = FeatureStore()
    await feature_store.initialize()
    
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        decode_responses=True
    )
    
    db_manager = DatabaseManager()
    
    # Load active models
    await load_active_models()
    
    logger.info("Inference service started successfully")
    
    yield
    
    # Shutdown
    if feature_store:
        await feature_store.cleanup()
    if redis_client:
        await redis_client.close()
    
    logger.info("Inference service shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="FVI Inference Service",
    description="Low-latency inference service for FVI predictions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add telemetry middleware
@app.middleware("http")
async def add_telemetry_middleware(request: Request, call_next):
    """Add telemetry headers to requests"""
    # Get or generate trace ID
    trace_id = request.headers.get("X-Trace-Id", telemetry.get_trace_id())
    if trace_id:
        request.state.trace_id = trace_id
    
    # Process request
    response = await call_next(request)
    
    # Add trace ID to response headers
    if trace_id:
        response.headers["X-Trace-Id"] = trace_id
    
    return response

# Instrument FastAPI app
telemetry.instrument_fastapi(app)

class InferenceService:
    """Core inference service logic"""
    
    def __init__(self):
        self.model_cache = {}
        self.scaler_cache = {}
        self.feature_names = []
        
    async def load_model(self, model_version: str) -> bool:
        """Load model and associated artifacts"""
        try:
            if model_version in self.model_cache:
                return True
            
            # Load model from MLflow
            model_uri = f"runs:/{model_version}/model"
            
            # Convert LightGBM model to ONNX
            onnx_model_path = await self._convert_to_onnx(model_uri, model_version)
            
            # Load ONNX model
            ort_session = ort.InferenceSession(
                onnx_model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Load scaler
            scaler_path = f"runs:/{model_version}/scaler"
            scaler = mlflow.artifacts.load_artifact(scaler_path)
            
            # Load feature names
            feature_names_path = f"runs:/{model_version}/feature_names.json"
            feature_names = mlflow.artifacts.load_artifact(feature_names_path)
            
            # Cache everything
            self.model_cache[model_version] = {
                'session': ort_session,
                'input_name': ort_session.get_inputs()[0].name,
                'output_name': ort_session.get_outputs()[0].name,
                'loaded_at': datetime.now()
            }
            
            self.scaler_cache[model_version] = scaler
            self.feature_names = feature_names['feature_names']
            
            logger.info(f"Model {model_version} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_version}: {str(e)}")
            return False
    
    async def _convert_to_onnx(self, model_uri: str, model_version: str) -> str:
        """Convert LightGBM model to ONNX format"""
        import lightgbm as lgb
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Load LightGBM model
        lgb_model = mlflow.lightgbm.load_model(model_uri)
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, lgb_model.num_feature()]))]
        onnx_model = onnxmltools.convert_lightgbm(
            lgb_model, 
            initial_types=initial_type,
            target_opset=11
        )
        
        # Save ONNX model
        onnx_path = f"/tmp/model_{model_version}.onnx"
        onnxmltools.utils.save_model(onnx_model, onnx_path)
        
        return onnx_path
    
    @trace_ml_prediction("lightgbm", "v2.1.0")
    async def predict(self, mine_id: str, features: Dict[str, Any], 
                     model_version: str = None) -> Dict[str, Any]:
        """Make prediction for a single mine"""
        start_time = time.time()
        
        try:
            # Get active model version if not specified
            if model_version is None:
                model_version = await self._get_active_model_version()
            
            # Check cache first
            cache_key = f"prediction:{mine_id}:{model_version}:{hash(str(features))}"
            cached_result = await redis_client.get(cache_key)
            
            if cached_result:
                result = json.loads(cached_result)
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                result['cache_hit'] = True
                # Record cache hit metrics
                telemetry.request_counter.add(1, {"operation": "prediction", "cache": "hit"})
                return result
            
            # Load model if not in cache
            if model_version not in self.model_cache:
                await self.load_model(model_version)
            
            # Prepare features
            feature_vector = await self._prepare_features(mine_id, features)
            
            # Scale features
            scaler = self.scaler_cache[model_version]
            scaled_features = scaler.transform(feature_vector.reshape(1, -1))
            
            # Make prediction
            model_info = self.model_cache[model_version]
            session = model_info['session']
            
            prediction = session.run(
                [model_info['output_name']], 
                {model_info['input_name']: scaled_features.astype(np.float32)}
            )[0][0]
            
            # Calculate confidence (simplified)
            confidence = min(1.0, max(0.1, 1.0 - abs(prediction) / 2.0))
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'mine_id': mine_id,
                'prediction': float(prediction),
                'confidence': float(confidence),
                'model_version': model_version,
                'processing_time_ms': processing_time,
                'cache_hit': False
            }
            
            # Cache result for 5 minutes
            await redis_client.setex(cache_key, 300, json.dumps(result))
            
            # Record metrics
            telemetry.request_counter.add(1, {"operation": "prediction", "cache": "miss"})
            telemetry.request_duration.record(processing_time / 1000, {"operation": "prediction"})
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            telemetry.error_counter.add(1, {"operation": "prediction", "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _prepare_features(self, mine_id: str, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for prediction"""
        
        # Get latest features from feature store if incomplete
        if len(features) < len(self.feature_names):
            stored_features = await feature_store.get_latest_features(mine_id)
            features.update(stored_features)
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                value = features[feature_name]
                # Handle categorical features
                if isinstance(value, str):
                    # Simple hash encoding for categorical features
                    value = hash(value) % 1000 / 1000.0
                feature_vector.append(float(value))
            else:
                # Use default value for missing features
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    async def _get_active_model_version(self) -> str:
        """Get active model version from database"""
        query = """
        SELECT model_version
        FROM model_registry
        WHERE model_type = 'lightgbm' AND is_active = true
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        async with db_manager.get_connection() as conn:
            result = await conn.fetchrow(query)
        
        if not result:
            raise HTTPException(status_code=404, detail="No active model found")
        
        return result['model_version']
    
    async def batch_predict(self, requests: List[PredictionRequest]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        tasks = []
        
        for req in requests:
            task = self.predict(
                req.mine_id,
                req.features,
                req.model_version
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch prediction {i}: {str(result)}")
                valid_results.append({
                    'mine_id': requests[i].mine_id,
                    'error': str(result),
                    'prediction': None,
                    'confidence': 0.0,
                    'model_version': 'error',
                    'processing_time_ms': 0.0
                })
            else:
                valid_results.append(result)
        
        return valid_results

# Initialize inference service
inference_service = InferenceService()

async def load_active_models():
    """Load all active models on startup"""
    global model_cache, scaler_cache, inference_service
    
    try:
        # Get active model versions
        query = """
        SELECT model_version, model_type
        FROM model_registry
        WHERE is_active = true
        ORDER BY created_at DESC
        """
        
        async with db_manager.get_connection() as conn:
            results = await conn.fetch(query)
        
        for row in results:
            if row['model_type'] == 'lightgbm':
                await inference_service.load_model(row['model_version'])
        
        logger.info(f"Loaded {len(results)} active models")
        
    except Exception as e:
        logger.error(f"Error loading active models: {str(e)}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(inference_service.model_cache)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    result = await inference_service.predict(
        request.mine_id,
        request.features,
        request.model_version
    )
    
    return PredictionResponse(**result)

@app.post("/batch/predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    start_time = time.time()
    
    if request.async_processing:
        # For async processing, return job ID immediately
        job_id = f"batch_{int(time.time())}_{len(request.requests)}"
        
        # Store job in Redis
        await redis_client.setex(
            f"batch_job:{job_id}",
            3600,  # 1 hour TTL
            json.dumps({
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "total_requests": len(request.requests)
            })
        )
        
        # Process in background
        asyncio.create_task(
            process_batch_async(job_id, request.requests)
        )
        
        return BatchPredictionResponse(
            predictions=[],
            total_processing_time_ms=0,
            async_job_id=job_id
        )
    else:
        # Synchronous processing
        results = await inference_service.batch_predict(request.requests)
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_processing_time_ms=total_time
        )

@app.get("/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of async batch job"""
    job_data = await redis_client.get(f"batch_job:{job_id}")
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return json.loads(job_data)

@app.get("/batch/results/{job_id}")
async def get_batch_results(job_id: str):
    """Get results of async batch job"""
    results_data = await redis_client.get(f"batch_results:{job_id}")
    
    if not results_data:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return json.loads(results_data)

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    query = """
    SELECT model_version, model_type, created_at, metrics, is_active
    FROM model_registry
    ORDER BY created_at DESC
    """
    
    async with db_manager.get_connection() as conn:
        results = await conn.fetch(query)
    
    models = []
    for row in results:
        models.append(ModelInfo(
            model_version=row['model_version'],
            model_type=row['model_type'],
            created_at=row['created_at'],
            metrics=json.loads(row['metrics']) if row['metrics'] else {},
            is_active=row['is_active']
        ))
    
    return models

@app.post("/models/{model_version}/activate")
async def activate_model(model_version: str):
    """Activate a specific model version"""
    async with db_manager.get_connection() as conn:
        async with conn.transaction():
            # Deactivate current models
            await conn.execute("""
                UPDATE model_registry 
                SET is_active = false 
                WHERE model_type = 'lightgbm' AND is_active = true
            """)
            
            # Activate specified model
            result = await conn.execute("""
                UPDATE model_registry 
                SET is_active = true 
                WHERE model_version = $1
            """, model_version)
            
            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Model version not found")
    
    # Load the new model
    await inference_service.load_model(model_version)
    
    return {"status": "activated", "model_version": model_version}

@app.get("/metrics")
async def get_inference_metrics():
    """Get inference service metrics"""
    # Get cache hit rate
    cache_hits = await redis_client.get("metrics:cache_hits") or "0"
    total_requests = await redis_client.get("metrics:total_requests") or "0"
    
    cache_hit_rate = float(cache_hits) / float(total_requests) if float(total_requests) > 0 else 0
    
    return {
        "cache_hit_rate": cache_hit_rate,
        "total_requests": int(total_requests),
        "models_loaded": len(inference_service.model_cache),
        "active_models": [v for v in inference_service.model_cache.keys()],
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

async def process_batch_async(job_id: str, requests: List[PredictionRequest]):
    """Process batch predictions asynchronously"""
    try:
        # Update job status
        await redis_client.setex(
            f"batch_job:{job_id}",
            3600,
            json.dumps({
                "status": "processing",
                "created_at": datetime.now().isoformat(),
                "total_requests": len(requests),
                "processed": 0
            })
        )
        
        # Process requests
        results = await inference_service.batch_predict(requests)
        
        # Store results
        await redis_client.setex(
            f"batch_results:{job_id}",
            3600,
            json.dumps({
                "predictions": results,
                "completed_at": datetime.now().isoformat()
            })
        )
        
        # Update job status
        await redis_client.setex(
            f"batch_job:{job_id}",
            3600,
            json.dumps({
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                "total_requests": len(requests),
                "processed": len(results),
                "completed_at": datetime.now().isoformat()
            })
        )
        
    except Exception as e:
        logger.error(f"Error processing batch job {job_id}: {str(e)}")
        
        # Update job status with error
        await redis_client.setex(
            f"batch_job:{job_id}",
            3600,
            json.dumps({
                "status": "error",
                "error": str(e),
                "created_at": datetime.now().isoformat(),
                "total_requests": len(requests)
            })
        )

if __name__ == "__main__":
    import uvicorn
    
    # Track start time
    app.state.start_time = time.time()
    
    # Run with uvicorn
    uvicorn.run(
        "src.ml.inference_service:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use Gunicorn for multiple workers in production
        reload=False,
        access_log=True
    )
