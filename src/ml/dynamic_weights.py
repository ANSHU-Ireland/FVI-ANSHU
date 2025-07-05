"""
Dynamic Weight Engine for FVI Analytics Platform

This module implements the dynamic weight calculation engine that:
1. Computes mutual information between metrics and FVI deltas
2. Updates weights based on information gain
3. Persists weights in Redis for low-latency access
4. Provides real-time weight retrieval for API endpoints

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, abs as spark_abs
from pyspark.sql.window import Window
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import config
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class DynamicWeightEngine:
    """Dynamic weight calculation and management engine"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.redis_client = None
        self.spark = None
        self.scaler = StandardScaler()
        
    async def initialize(self) -> None:
        """Initialize connections and resources"""
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("FVI_DynamicWeightEngine") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        logger.info("Dynamic weight engine initialized successfully")
    
    async def calculate_information_gain(self, lookback_days: int = 7) -> Dict[str, float]:
        """
        Calculate mutual information between metrics and FVI deltas
        
        Args:
            lookback_days: Number of days to look back for analysis
            
        Returns:
            Dictionary of metric_slug -> information_gain_score
        """
        try:
            # Get FVI historical data with deltas
            fvi_data = await self._get_fvi_historical_data(lookback_days)
            
            if fvi_data.empty:
                logger.warning("No FVI historical data available")
                return {}
            
            # Get feature data
            feature_data = await self._get_feature_data(lookback_days)
            
            if feature_data.empty:
                logger.warning("No feature data available")
                return {}
            
            # Merge and prepare data
            merged_data = self._prepare_data_for_analysis(fvi_data, feature_data)
            
            # Calculate mutual information
            information_gains = self._calculate_mutual_information(merged_data)
            
            logger.info(f"Calculated information gains for {len(information_gains)} metrics")
            return information_gains
            
        except Exception as e:
            logger.error(f"Error calculating information gain: {str(e)}")
            return {}
    
    async def _get_fvi_historical_data(self, lookback_days: int) -> pd.DataFrame:
        """Get FVI historical data with calculated deltas"""
        
        # Use Spark for large dataset processing
        spark_df = self.spark.read.format("jdbc") \
            .option("url", f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}") \
            .option("dbtable", "gold_mine_features") \
            .option("user", config.DB_USER) \
            .option("password", config.DB_PASSWORD) \
            .load()
        
        # Filter for recent data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        spark_df = spark_df.filter(col("feature_date") >= cutoff_date)
        
        # Calculate FVI deltas using window functions
        window_spec = Window.partitionBy("mine_id").orderBy("feature_date")
        
        spark_df = spark_df.withColumn(
            "prev_fvi_score", 
            lag("feature_freshness_score", 1).over(window_spec)
        ).withColumn(
            "fvi_delta",
            col("feature_freshness_score") - col("prev_fvi_score")
        ).withColumn(
            "fvi_delta_abs",
            spark_abs(col("fvi_delta"))
        )
        
        # Convert to pandas for sklearn processing
        pandas_df = spark_df.toPandas()
        
        return pandas_df
    
    async def _get_feature_data(self, lookback_days: int) -> pd.DataFrame:
        """Get feature data for analysis"""
        
        query = """
        SELECT 
            mine_id,
            feature_date,
            production_mt,
            capacity_utilization_rate,
            esg_score,
            carbon_intensity_tco2_per_mt,
            revenue_per_mt,
            operating_margin,
            safety_incident_rate_per_1000_employees,
            reserve_life_years,
            total_carbon_emissions_tco2,
            productivity_mt_per_employee,
            latitude_abs,
            metric_year,
            metric_quarter,
            metric_month
        FROM gold_mine_features 
        WHERE feature_date >= %s
        AND feature_freshness_score >= 0.7
        ORDER BY mine_id, feature_date
        """
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        async with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[cutoff_date])
        
        return df
    
    def _prepare_data_for_analysis(self, fvi_data: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and merge data for mutual information analysis"""
        
        # Merge datasets
        merged = pd.merge(
            fvi_data[['mine_id', 'feature_date', 'fvi_delta', 'fvi_delta_abs']],
            feature_data,
            on=['mine_id', 'feature_date'],
            how='inner'
        )
        
        # Remove rows with null FVI deltas (first observations)
        merged = merged.dropna(subset=['fvi_delta'])
        
        # Fill missing values with median
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].median())
        
        return merged
    
    def _calculate_mutual_information(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate mutual information between features and FVI deltas"""
        
        # Feature columns to analyze
        feature_cols = [
            'production_mt', 'capacity_utilization_rate', 'esg_score',
            'carbon_intensity_tco2_per_mt', 'revenue_per_mt', 'operating_margin',
            'safety_incident_rate_per_1000_employees', 'reserve_life_years',
            'total_carbon_emissions_tco2', 'productivity_mt_per_employee',
            'latitude_abs', 'metric_year', 'metric_quarter', 'metric_month'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if not available_cols:
            logger.warning("No feature columns available for analysis")
            return {}
        
        # Prepare feature matrix
        X = data[available_cols].values
        y = data['fvi_delta_abs'].values  # Use absolute delta as target
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
        
        # Create metric mapping
        metric_mapping = {
            'production_mt': 'production_volume',
            'capacity_utilization_rate': 'operational_efficiency',
            'esg_score': 'sustainability_score',
            'carbon_intensity_tco2_per_mt': 'carbon_footprint',
            'revenue_per_mt': 'profitability',
            'operating_margin': 'cost_efficiency',
            'safety_incident_rate_per_1000_employees': 'safety_performance',
            'reserve_life_years': 'resource_longevity',
            'total_carbon_emissions_tco2': 'environmental_impact',
            'productivity_mt_per_employee': 'labor_productivity',
            'latitude_abs': 'geographic_risk',
            'metric_year': 'temporal_trend',
            'metric_quarter': 'seasonal_pattern',
            'metric_month': 'monthly_cycle'
        }
        
        # Normalize scores to [0, 1] range
        max_score = max(mi_scores) if mi_scores.max() > 0 else 1
        normalized_scores = mi_scores / max_score
        
        # Build result dictionary
        information_gains = {}
        for i, col in enumerate(available_cols):
            metric_slug = metric_mapping.get(col, col)
            information_gains[metric_slug] = float(normalized_scores[i])
        
        return information_gains
    
    async def update_weights(self, information_gains: Dict[str, float]) -> None:
        """Update weights based on information gain scores"""
        
        try:
            # Get current weights from database
            current_weights = await self._get_current_weights()
            
            # Calculate new weights with decay factor
            decay_factor = 0.8  # Blend with historical weights
            new_weights = {}
            
            for metric_slug, info_gain in information_gains.items():
                current_weight = current_weights.get(metric_slug, 0.5)
                
                # Apply exponential moving average
                new_weight = decay_factor * current_weight + (1 - decay_factor) * info_gain
                new_weights[metric_slug] = max(0.1, min(1.0, new_weight))  # Clamp to [0.1, 1.0]
            
            # Update database
            await self._update_weights_in_db(new_weights)
            
            # Update Redis cache
            await self._update_weights_in_redis(new_weights)
            
            logger.info(f"Updated weights for {len(new_weights)} metrics")
            
        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")
            raise
    
    async def _get_current_weights(self) -> Dict[str, float]:
        """Get current weights from database"""
        
        query = """
        SELECT metric_slug, weight_value
        FROM dim_weight
        WHERE is_active = true
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.fetch(query)
        
        return {row['metric_slug']: row['weight_value'] for row in result}
    
    async def _update_weights_in_db(self, weights: Dict[str, float]) -> None:
        """Update weights in database"""
        
        timestamp = datetime.now()
        
        async with self.db_manager.get_connection() as conn:
            async with conn.transaction():
                # Deactivate current weights
                await conn.execute("""
                    UPDATE dim_weight 
                    SET is_active = false, updated_at = $1
                    WHERE is_active = true
                """, timestamp)
                
                # Insert new weights
                for metric_slug, weight_value in weights.items():
                    await conn.execute("""
                        INSERT INTO dim_weight (
                            metric_slug, weight_value, weight_type, 
                            is_active, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """, metric_slug, weight_value, 'DYNAMIC', True, timestamp, timestamp)
    
    async def _update_weights_in_redis(self, weights: Dict[str, float]) -> None:
        """Update weights in Redis cache"""
        
        # Store individual weights
        pipe = self.redis_client.pipeline()
        
        for metric_slug, weight_value in weights.items():
            key = f"weight:{metric_slug}"
            pipe.setex(key, 24 * 3600, str(weight_value))  # 24 hour TTL
        
        # Store complete weights dictionary
        weights_json = json.dumps(weights)
        pipe.setex("weights:all", 24 * 3600, weights_json)
        
        # Store metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "total_metrics": len(weights),
            "avg_weight": np.mean(list(weights.values())),
            "min_weight": min(weights.values()),
            "max_weight": max(weights.values())
        }
        pipe.setex("weights:metadata", 24 * 3600, json.dumps(metadata))
        
        await pipe.execute()
    
    async def get_weights(self, metric_slugs: Optional[List[str]] = None) -> Dict[str, float]:
        """Get weights from Redis cache with database fallback"""
        
        try:
            # Try Redis first
            if metric_slugs:
                weights = {}
                for slug in metric_slugs:
                    weight_str = await self.redis_client.get(f"weight:{slug}")
                    if weight_str:
                        weights[slug] = float(weight_str)
                    else:
                        # Fallback to database
                        db_weight = await self._get_weight_from_db(slug)
                        weights[slug] = db_weight if db_weight is not None else 0.5
                return weights
            else:
                # Get all weights
                weights_json = await self.redis_client.get("weights:all")
                if weights_json:
                    return json.loads(weights_json)
                else:
                    # Fallback to database
                    return await self._get_current_weights()
                    
        except Exception as e:
            logger.error(f"Error getting weights from Redis: {str(e)}")
            # Fallback to database
            if metric_slugs:
                weights = {}
                for slug in metric_slugs:
                    db_weight = await self._get_weight_from_db(slug)
                    weights[slug] = db_weight if db_weight is not None else 0.5
                return weights
            else:
                return await self._get_current_weights()
    
    async def _get_weight_from_db(self, metric_slug: str) -> Optional[float]:
        """Get single weight from database"""
        
        query = """
        SELECT weight_value
        FROM dim_weight
        WHERE metric_slug = $1 AND is_active = true
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.fetchrow(query, metric_slug)
        
        return result['weight_value'] if result else None
    
    async def run_nightly_update(self) -> Dict[str, any]:
        """Run nightly weight update process"""
        
        start_time = datetime.now()
        logger.info("Starting nightly weight update process")
        
        try:
            # Calculate information gains
            information_gains = await self.calculate_information_gain(lookback_days=7)
            
            if not information_gains:
                logger.warning("No information gains calculated, skipping weight update")
                return {
                    "status": "skipped",
                    "reason": "no_information_gains",
                    "duration_seconds": (datetime.now() - start_time).total_seconds()
                }
            
            # Update weights
            await self.update_weights(information_gains)
            
            # Generate report
            report = await self._generate_weight_report(information_gains)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Nightly weight update completed in {duration:.2f} seconds")
            
            return {
                "status": "completed",
                "duration_seconds": duration,
                "metrics_updated": len(information_gains),
                "report": report,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in nightly weight update: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    async def _generate_weight_report(self, information_gains: Dict[str, float]) -> Dict[str, any]:
        """Generate weight update report"""
        
        current_weights = await self._get_current_weights()
        
        # Calculate changes
        changes = {}
        for metric_slug, new_gain in information_gains.items():
            old_weight = current_weights.get(metric_slug, 0.5)
            change = new_gain - old_weight
            changes[metric_slug] = {
                "old_weight": old_weight,
                "new_weight": new_gain,
                "change": change,
                "change_pct": (change / old_weight * 100) if old_weight > 0 else 0
            }
        
        # Top movers
        top_increases = sorted(changes.items(), key=lambda x: x[1]["change"], reverse=True)[:5]
        top_decreases = sorted(changes.items(), key=lambda x: x[1]["change"])[:5]
        
        return {
            "total_metrics": len(information_gains),
            "avg_weight": np.mean(list(information_gains.values())),
            "top_increases": [{"metric": k, **v} for k, v in top_increases],
            "top_decreases": [{"metric": k, **v} for k, v in top_decreases],
            "summary_stats": {
                "min_weight": min(information_gains.values()),
                "max_weight": max(information_gains.values()),
                "std_weight": np.std(list(information_gains.values()))
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.spark:
            self.spark.stop()
        logger.info("Dynamic weight engine cleanup completed")


# Async context manager
class AsyncDynamicWeightEngine:
    """Async context manager for DynamicWeightEngine"""
    
    def __init__(self):
        self.engine = DynamicWeightEngine()
    
    async def __aenter__(self):
        await self.engine.initialize()
        return self.engine
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.engine.cleanup()
