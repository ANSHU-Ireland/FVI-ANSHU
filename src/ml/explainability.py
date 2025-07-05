"""
SHAP Explainability Cache System for FVI Analytics Platform

This module implements the SHAP explainability system with:
1. Pre-computed SHAP values for top entities stored in Redis
2. On-the-fly SHAP computation for cold entities
3. Server-sent events for streaming explanations
4. Compressed NumPy arrays for efficient storage
5. Fallback logic for high-availability

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
import pickle
import gzip
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import redis.asyncio as redis
import shap
import lightgbm as lgb
import mlflow
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import config
from src.database.db_manager import DatabaseManager
from src.ml.feature_store import FeatureStore
from src.ml.inference_service import InferenceService

logger = logging.getLogger(__name__)

class SHAPExplainabilityCache:
    """SHAP explainability cache with pre-computation and streaming"""
    
    def __init__(self):
        self.redis_client = None
        self.db_manager = DatabaseManager()
        self.feature_store = FeatureStore()
        self.inference_service = InferenceService()
        self.explainers = {}  # Model version -> SHAP explainer
        self.models = {}      # Model version -> Actual model
        self.scalers = {}     # Model version -> Scaler
        self.feature_names = []
        
    async def initialize(self):
        """Initialize the explainability cache system"""
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=False  # We'll handle binary data
        )
        
        # Initialize other services
        await self.feature_store.initialize()
        
        # Load active models and create explainers
        await self.load_active_models()
        
        logger.info("SHAP explainability cache initialized")
    
    async def load_active_models(self):
        """Load active models and create SHAP explainers"""
        try:
            # Get active model versions
            query = """
            SELECT model_version, model_type
            FROM model_registry
            WHERE is_active = true AND model_type = 'lightgbm'
            ORDER BY created_at DESC
            """
            
            async with self.db_manager.get_connection() as conn:
                results = await conn.fetch(query)
            
            for row in results:
                model_version = row['model_version']
                await self.create_explainer(model_version)
            
            logger.info(f"Created explainers for {len(results)} models")
            
        except Exception as e:
            logger.error(f"Error loading models for explainability: {str(e)}")
    
    async def create_explainer(self, model_version: str):
        """Create SHAP explainer for a model version"""
        try:
            # Load model artifacts
            model_uri = f"runs:/{model_version}/model"
            model = mlflow.lightgbm.load_model(model_uri)
            
            # Load scaler
            scaler_uri = f"runs:/{model_version}/scaler"
            scaler = mlflow.artifacts.load_artifact(scaler_uri)
            
            # Load feature names
            feature_names_uri = f"runs:/{model_version}/feature_names.json"
            feature_names = mlflow.artifacts.load_artifact(feature_names_uri)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Store in cache
            self.explainers[model_version] = explainer
            self.models[model_version] = model
            self.scalers[model_version] = scaler
            self.feature_names = feature_names['feature_names']
            
            logger.info(f"Created SHAP explainer for model {model_version}")
            
        except Exception as e:
            logger.error(f"Error creating explainer for model {model_version}: {str(e)}")
    
    async def precompute_shap_values(self, model_version: str = None, top_n: int = 1000):
        """
        Pre-compute SHAP values for top-N entities
        
        Args:
            model_version: Model version to use (if None, use active model)
            top_n: Number of top entities to pre-compute
        """
        start_time = time.time()
        logger.info(f"Starting SHAP pre-computation for top {top_n} entities")
        
        try:
            if model_version is None:
                model_version = await self._get_active_model_version()
            
            if model_version not in self.explainers:
                await self.create_explainer(model_version)
            
            explainer = self.explainers[model_version]
            scaler = self.scalers[model_version]
            
            # Get top entities based on prediction frequency or importance
            top_entities = await self._get_top_entities(top_n)
            
            if not top_entities:
                logger.warning("No entities found for SHAP pre-computation")
                return
            
            # Get features for top entities
            features_data = await self._get_entities_features(top_entities)
            
            # Prepare feature matrix
            X = self._prepare_feature_matrix(features_data)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Calculate SHAP values in batches
            batch_size = 50
            total_computed = 0
            
            for i in range(0, len(X_scaled), batch_size):
                batch_X = X_scaled[i:i+batch_size]
                batch_entities = top_entities[i:i+batch_size]
                
                # Compute SHAP values
                shap_values = explainer.shap_values(batch_X)
                
                # Store in Redis
                for j, mine_id in enumerate(batch_entities):
                    await self._store_shap_values(
                        mine_id=mine_id,
                        model_version=model_version,
                        shap_values=shap_values[j],
                        base_value=explainer.expected_value,
                        feature_names=self.feature_names
                    )
                
                total_computed += len(batch_entities)
                
                # Log progress
                if total_computed % 100 == 0:
                    logger.info(f"Pre-computed SHAP values for {total_computed}/{len(top_entities)} entities")
            
            duration = time.time() - start_time
            logger.info(f"SHAP pre-computation completed in {duration:.2f}s. Computed {total_computed} entities")
            
            # Store metadata
            await self._store_precomputation_metadata(model_version, total_computed, duration)
            
        except Exception as e:
            logger.error(f"Error in SHAP pre-computation: {str(e)}")
            raise
    
    async def get_shap_explanation(self, mine_id: str, features: Dict[str, Any] = None,
                                  model_version: str = None) -> Dict[str, Any]:
        """
        Get SHAP explanation for a specific entity
        
        Args:
            mine_id: Mine identifier
            features: Feature values (if None, will fetch from feature store)
            model_version: Model version to use
            
        Returns:
            SHAP explanation dictionary
        """
        try:
            if model_version is None:
                model_version = await self._get_active_model_version()
            
            # Check cache first
            cache_key = f"shap:{model_version}:{mine_id}"
            cached_shap = await self.redis_client.get(cache_key)
            
            if cached_shap:
                # Decompress and deserialize
                shap_data = pickle.loads(gzip.decompress(cached_shap))
                shap_data['cache_hit'] = True
                return shap_data
            
            # Compute on-the-fly
            logger.info(f"Computing SHAP values on-the-fly for {mine_id}")
            
            if model_version not in self.explainers:
                await self.create_explainer(model_version)
            
            explainer = self.explainers[model_version]
            scaler = self.scalers[model_version]
            
            # Get or prepare features
            if features is None:
                features = await self.feature_store.get_latest_features(mine_id)
            
            # Prepare feature vector
            feature_vector = self._prepare_single_feature_vector(features)
            
            # Scale features
            X_scaled = scaler.transform(feature_vector.reshape(1, -1))
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_scaled[0])
            
            # Create explanation
            explanation = {
                'mine_id': mine_id,
                'model_version': model_version,
                'shap_values': shap_values.tolist(),
                'base_value': float(explainer.expected_value),
                'feature_names': self.feature_names,
                'feature_values': feature_vector.tolist(),
                'computed_at': datetime.now().isoformat(),
                'cache_hit': False
            }
            
            # Cache for future use (1 hour TTL)
            await self._store_shap_values(
                mine_id=mine_id,
                model_version=model_version,
                shap_values=shap_values,
                base_value=explainer.expected_value,
                feature_names=self.feature_names,
                ttl=3600
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error getting SHAP explanation for {mine_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_shap_explanation(self, mine_id: str, features: Dict[str, Any] = None,
                                     model_version: str = None) -> AsyncGenerator[str, None]:
        """
        Stream SHAP explanation computation progress
        
        Args:
            mine_id: Mine identifier
            features: Feature values
            model_version: Model version to use
            
        Yields:
            Server-sent events with computation progress
        """
        try:
            yield f"data: {json.dumps({'status': 'started', 'mine_id': mine_id})}\n\n"
            
            if model_version is None:
                model_version = await self._get_active_model_version()
            
            # Check cache first
            cache_key = f"shap:{model_version}:{mine_id}"
            cached_shap = await self.redis_client.get(cache_key)
            
            if cached_shap:
                shap_data = pickle.loads(gzip.decompress(cached_shap))
                shap_data['cache_hit'] = True
                yield f"data: {json.dumps({'status': 'completed', 'explanation': shap_data})}\n\n"
                return
            
            yield f"data: {json.dumps({'status': 'computing', 'message': 'Loading model and features'})}\n\n"
            
            # Load model if needed
            if model_version not in self.explainers:
                await self.create_explainer(model_version)
            
            explainer = self.explainers[model_version]
            scaler = self.scalers[model_version]
            
            # Get features
            if features is None:
                yield f"data: {json.dumps({'status': 'computing', 'message': 'Fetching features from feature store'})}\n\n"
                features = await self.feature_store.get_latest_features(mine_id)
            
            yield f"data: {json.dumps({'status': 'computing', 'message': 'Preparing feature vector'})}\n\n"
            
            # Prepare and scale features
            feature_vector = self._prepare_single_feature_vector(features)
            X_scaled = scaler.transform(feature_vector.reshape(1, -1))
            
            yield f"data: {json.dumps({'status': 'computing', 'message': 'Computing SHAP values'})}\n\n"
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_scaled[0])
            
            # Create explanation
            explanation = {
                'mine_id': mine_id,
                'model_version': model_version,
                'shap_values': shap_values.tolist(),
                'base_value': float(explainer.expected_value),
                'feature_names': self.feature_names,
                'feature_values': feature_vector.tolist(),
                'computed_at': datetime.now().isoformat(),
                'cache_hit': False
            }
            
            # Cache for future use
            await self._store_shap_values(
                mine_id=mine_id,
                model_version=model_version,
                shap_values=shap_values,
                base_value=explainer.expected_value,
                feature_names=self.feature_names,
                ttl=3600
            )
            
            yield f"data: {json.dumps({'status': 'completed', 'explanation': explanation})}\n\n"
            
        except Exception as e:
            logger.error(f"Error streaming SHAP explanation for {mine_id}: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    
    async def _get_top_entities(self, top_n: int) -> List[str]:
        """Get top entities for pre-computation"""
        
        # Get entities with highest prediction frequency or importance
        query = """
        SELECT mine_id, COUNT(*) as prediction_count
        FROM (
            SELECT DISTINCT mine_id, feature_date
            FROM gold_mine_features
            WHERE feature_date >= NOW() - INTERVAL '30 days'
            AND feature_freshness_score >= 0.7
        ) t
        GROUP BY mine_id
        ORDER BY prediction_count DESC, mine_id
        LIMIT %s
        """
        
        async with self.db_manager.get_connection() as conn:
            results = await conn.fetch(query, top_n)
        
        return [row['mine_id'] for row in results]
    
    async def _get_entities_features(self, mine_ids: List[str]) -> pd.DataFrame:
        """Get latest features for multiple entities"""
        
        # Use feature store to get latest features
        all_features = []
        
        for mine_id in mine_ids:
            try:
                features = await self.feature_store.get_latest_features(mine_id)
                features['mine_id'] = mine_id
                all_features.append(features)
            except Exception as e:
                logger.warning(f"Error getting features for {mine_id}: {str(e)}")
                continue
        
        return pd.DataFrame(all_features)
    
    def _prepare_feature_matrix(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for SHAP computation"""
        
        # Ensure feature columns are in correct order
        feature_vectors = []
        
        for _, row in features_df.iterrows():
            vector = []
            for feature_name in self.feature_names:
                if feature_name in row:
                    value = row[feature_name]
                    # Handle categorical features
                    if isinstance(value, str):
                        value = hash(value) % 1000 / 1000.0
                    vector.append(float(value))
                else:
                    vector.append(0.0)
            feature_vectors.append(vector)
        
        return np.array(feature_vectors)
    
    def _prepare_single_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare single feature vector"""
        
        vector = []
        for feature_name in self.feature_names:
            if feature_name in features:
                value = features[feature_name]
                # Handle categorical features
                if isinstance(value, str):
                    value = hash(value) % 1000 / 1000.0
                vector.append(float(value))
            else:
                vector.append(0.0)
        
        return np.array(vector)
    
    async def _store_shap_values(self, mine_id: str, model_version: str, shap_values: np.ndarray,
                                base_value: float, feature_names: List[str], ttl: int = 86400):
        """Store SHAP values in Redis with compression"""
        
        # Create SHAP data dictionary
        shap_data = {
            'mine_id': mine_id,
            'model_version': model_version,
            'shap_values': shap_values.tolist(),
            'base_value': float(base_value),
            'feature_names': feature_names,
            'computed_at': datetime.now().isoformat(),
            'cache_hit': False
        }
        
        # Serialize and compress
        serialized = pickle.dumps(shap_data)
        compressed = gzip.compress(serialized)
        
        # Store in Redis
        cache_key = f"shap:{model_version}:{mine_id}"
        await self.redis_client.setex(cache_key, ttl, compressed)
    
    async def _store_precomputation_metadata(self, model_version: str, total_computed: int, duration: float):
        """Store pre-computation metadata"""
        
        metadata = {
            'model_version': model_version,
            'total_computed': total_computed,
            'duration_seconds': duration,
            'computed_at': datetime.now().isoformat()
        }
        
        # Store in Redis
        metadata_key = f"shap_metadata:{model_version}"
        await self.redis_client.setex(metadata_key, 86400, json.dumps(metadata))
    
    async def _get_active_model_version(self) -> str:
        """Get active model version"""
        query = """
        SELECT model_version
        FROM model_registry
        WHERE model_type = 'lightgbm' AND is_active = true
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchrow(query)
        
        if not result:
            raise ValueError("No active model found")
        
        return result['model_version']
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            # Get cache keys
            shap_keys = await self.redis_client.keys("shap:*")
            metadata_keys = await self.redis_client.keys("shap_metadata:*")
            
            # Calculate cache size
            total_size = 0
            for key in shap_keys:
                size = await self.redis_client.memory_usage(key)
                if size:
                    total_size += size
            
            # Get metadata
            metadata = {}
            for key in metadata_keys:
                data = await self.redis_client.get(key)
                if data:
                    metadata[key] = json.loads(data)
            
            return {
                'total_cached_explanations': len(shap_keys),
                'cache_size_bytes': total_size,
                'cache_size_mb': total_size / (1024 * 1024),
                'precomputation_metadata': metadata,
                'explainers_loaded': len(self.explainers)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': str(e)}
    
    async def clear_cache(self, model_version: str = None):
        """Clear SHAP cache"""
        try:
            if model_version:
                # Clear specific model cache
                pattern = f"shap:{model_version}:*"
                keys = await self.redis_client.keys(pattern)
                
                if keys:
                    await self.redis_client.delete(*keys)
                
                # Clear metadata
                metadata_key = f"shap_metadata:{model_version}"
                await self.redis_client.delete(metadata_key)
                
                logger.info(f"Cleared {len(keys)} SHAP cache entries for model {model_version}")
            else:
                # Clear all SHAP cache
                shap_keys = await self.redis_client.keys("shap:*")
                metadata_keys = await self.redis_client.keys("shap_metadata:*")
                
                all_keys = shap_keys + metadata_keys
                if all_keys:
                    await self.redis_client.delete(*all_keys)
                
                logger.info(f"Cleared {len(all_keys)} SHAP cache entries")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise
    
    async def run_nightly_precomputation(self, top_n: int = 1000) -> Dict[str, Any]:
        """Run nightly SHAP pre-computation"""
        start_time = datetime.now()
        logger.info("Starting nightly SHAP pre-computation")
        
        try:
            # Get active model
            model_version = await self._get_active_model_version()
            
            # Pre-compute SHAP values
            await self.precompute_shap_values(model_version, top_n)
            
            # Get cache stats
            stats = await self.get_cache_stats()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'completed',
                'model_version': model_version,
                'duration_seconds': duration,
                'cache_stats': stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in nightly SHAP pre-computation: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    async def cleanup(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.feature_store:
            await self.feature_store.cleanup()
        logger.info("SHAP explainability cache cleanup completed")


# FastAPI routes for SHAP explainability
def create_shap_routes(app: FastAPI):
    """Create SHAP explainability routes"""
    
    shap_cache = SHAPExplainabilityCache()
    
    @app.on_event("startup")
    async def startup_shap():
        await shap_cache.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_shap():
        await shap_cache.cleanup()
    
    @app.get("/explain/{mine_id}")
    async def get_explanation(mine_id: str, model_version: str = None):
        """Get SHAP explanation for a mine"""
        return await shap_cache.get_shap_explanation(mine_id, model_version=model_version)
    
    @app.get("/explain/{mine_id}/stream")
    async def stream_explanation(mine_id: str, model_version: str = None):
        """Stream SHAP explanation computation"""
        return EventSourceResponse(
            shap_cache.stream_shap_explanation(mine_id, model_version=model_version)
        )
    
    @app.post("/explain/precompute")
    async def precompute_explanations(top_n: int = 1000, model_version: str = None):
        """Trigger SHAP pre-computation"""
        await shap_cache.precompute_shap_values(model_version, top_n)
        return {"status": "started", "top_n": top_n}
    
    @app.get("/explain/cache/stats")
    async def get_cache_stats():
        """Get cache statistics"""
        return await shap_cache.get_cache_stats()
    
    @app.delete("/explain/cache")
    async def clear_cache(model_version: str = None):
        """Clear SHAP cache"""
        await shap_cache.clear_cache(model_version)
        return {"status": "cleared"}
    
    return shap_cache

if __name__ == "__main__":
    # Example usage
    async def main():
        cache = SHAPExplainabilityCache()
        await cache.initialize()
        
        try:
            # Pre-compute for top 100 entities
            await cache.precompute_shap_values(top_n=100)
            
            # Get explanation for a specific mine
            explanation = await cache.get_shap_explanation("MINE_001")
            print(json.dumps(explanation, indent=2))
            
        finally:
            await cache.cleanup()
    
    asyncio.run(main())
