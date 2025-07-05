"""
ML Training Pipeline for FVI Analytics Platform

This module implements the machine learning training pipeline with:
1. Feature store hydration and temporal splitting
2. Optuna-driven hyperparameter optimization
3. LightGBM and PyMC Bayesian hierarchical models
4. MLflow experiment tracking and model versioning
5. Model artifacts with full environment lockfiles

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
import pickle
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import optuna
import pymc as pm
import arviz as az
import mlflow
import mlflow.lightgbm
import mlflow.pytorch
from mlflow.models import infer_signature
import joblib
import yaml

from src.config import config
from src.database.db_manager import DatabaseManager
from src.ml.feature_store import FeatureStore
from src.ml.dynamic_weights import DynamicWeightEngine

logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Machine learning training pipeline with MLflow integration"""
    
    def __init__(self, experiment_name: str = "fvi_prediction"):
        self.experiment_name = experiment_name
        self.db_manager = DatabaseManager()
        self.feature_store = FeatureStore()
        self.weight_engine = DynamicWeightEngine()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # MLflow setup
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        
    async def initialize(self) -> None:
        """Initialize pipeline components"""
        await self.feature_store.initialize()
        await self.weight_engine.initialize()
        logger.info("ML training pipeline initialized")
    
    async def hydrate_feature_store(self, 
                                   start_date: datetime = None,
                                   end_date: datetime = None) -> pd.DataFrame:
        """
        Hydrate feature store snapshot for training
        
        Args:
            start_date: Start date for feature extraction
            end_date: End date for feature extraction
            
        Returns:
            DataFrame with features and target variables
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)  # 1 year lookback
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Hydrating feature store from {start_date} to {end_date}")
        
        # Get features from feature store
        features = await self.feature_store.get_historical_features(
            start_date=start_date,
            end_date=end_date
        )
        
        # Get target variable (FVI deltas)
        targets = await self._get_target_variable(start_date, end_date)
        
        # Merge features and targets
        data = pd.merge(
            features, 
            targets, 
            on=['mine_id', 'feature_date'], 
            how='inner'
        )
        
        logger.info(f"Hydrated {len(data)} feature records")
        return data
    
    async def _get_target_variable(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get target variable (FVI deltas) for training"""
        
        query = """
        WITH fvi_with_lag AS (
            SELECT 
                mine_id,
                feature_date,
                feature_freshness_score as fvi_score,
                LAG(feature_freshness_score, 1) OVER (
                    PARTITION BY mine_id 
                    ORDER BY feature_date
                ) as prev_fvi_score
            FROM gold_mine_features
            WHERE feature_date BETWEEN %s AND %s
            AND feature_freshness_score IS NOT NULL
        )
        SELECT 
            mine_id,
            feature_date,
            fvi_score,
            prev_fvi_score,
            fvi_score - prev_fvi_score as fvi_delta,
            ABS(fvi_score - prev_fvi_score) as fvi_delta_abs,
            CASE 
                WHEN fvi_score - prev_fvi_score > 0.1 THEN 'INCREASE'
                WHEN fvi_score - prev_fvi_score < -0.1 THEN 'DECREASE'
                ELSE 'STABLE'
            END as fvi_direction
        FROM fvi_with_lag
        WHERE prev_fvi_score IS NOT NULL
        ORDER BY mine_id, feature_date
        """
        
        async with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training with temporal splits
        
        Args:
            data: Combined features and target data
            
        Returns:
            Tuple of (features, targets) DataFrames
        """
        # Feature columns
        feature_cols = [
            'production_mt', 'capacity_utilization_rate', 'esg_score',
            'carbon_intensity_tco2_per_mt', 'revenue_per_mt', 'operating_margin',
            'safety_incident_rate_per_1000_employees', 'reserve_life_years',
            'total_carbon_emissions_tco2', 'productivity_mt_per_employee',
            'latitude_abs', 'metric_year', 'metric_quarter', 'metric_month'
        ]
        
        # Filter available columns
        available_feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Add categorical features
        categorical_cols = [
            'mine_type_standardized', 'operational_status_standardized',
            'country_standardized', 'utilization_category', 'esg_category',
            'carbon_intensity_category'
        ]
        
        available_categorical_cols = [col for col in categorical_cols if col in data.columns]
        
        # Prepare features
        X = data[available_feature_cols + available_categorical_cols + ['mine_id', 'feature_date']].copy()
        
        # Encode categorical features
        for col in available_categorical_cols:
            X[f'{col}_encoded'] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # Prepare targets
        y = data[['mine_id', 'feature_date', 'fvi_delta', 'fvi_delta_abs', 'fvi_direction']].copy()
        
        # Remove rows with missing critical values
        mask = ~(X[available_feature_cols].isnull().any(axis=1) | y[['fvi_delta', 'fvi_delta_abs']].isnull().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} training samples with {len(available_feature_cols)} features")
        
        return X, y
    
    def create_temporal_splits(self, X: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5) -> List[Tuple]:
        """
        Create temporal splits for time series cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            n_splits: Number of temporal splits
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Sort by date
        combined = pd.concat([X, y[['fvi_delta']]], axis=1)
        combined = combined.sort_values('feature_date')
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(combined):
            splits.append((train_idx, val_idx))
        
        logger.info(f"Created {len(splits)} temporal splits")
        return splits
    
    def optimize_lightgbm_params(self, X: pd.DataFrame, y: pd.DataFrame, 
                                 n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters using Optuna
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Prepare data for LightGBM
        feature_cols = [col for col in X.columns if col not in ['mine_id', 'feature_date']]
        X_lgb = X[feature_cols].copy()
        y_lgb = y['fvi_delta_abs'].copy()  # Use absolute delta for regression
        
        # Create temporal splits
        splits = self.create_temporal_splits(X, y, n_splits=3)
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'verbosity': -1,
                'seed': 42
            }
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in splits:
                X_train_fold = X_lgb.iloc[train_idx]
                y_train_fold = y_lgb.iloc[train_idx]
                X_val_fold = X_lgb.iloc[val_idx]
                y_val_fold = y_lgb.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                # Train model
                train_data = lgb.Dataset(X_train_scaled, label=y_train_fold)
                val_data = lgb.Dataset(X_val_scaled, label=y_val_fold, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                # Predict and score
                y_pred = model.predict(X_val_scaled)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(rmse)
            
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Optimization completed. Best RMSE: {study.best_value:.4f}")
        
        return study.best_params
    
    async def train_lightgbm_model(self, X: pd.DataFrame, y: pd.DataFrame,
                                   params: Dict[str, Any] = None) -> Tuple[lgb.Booster, Dict[str, float]]:
        """
        Train LightGBM model with best parameters
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            params: Model parameters (if None, will optimize)
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info("Training LightGBM model")
        
        # Optimize parameters if not provided
        if params is None:
            params = self.optimize_lightgbm_params(X, y)
        
        # Prepare data
        feature_cols = [col for col in X.columns if col not in ['mine_id', 'feature_date']]
        X_lgb = X[feature_cols].copy()
        y_lgb = y['fvi_delta_abs'].copy()
        
        # Split data temporally (80/20 split)
        split_date = X['feature_date'].quantile(0.8)
        train_mask = X['feature_date'] <= split_date
        
        X_train = X_lgb[train_mask]
        X_val = X_lgb[~train_mask]
        y_train = y_lgb[train_mask]
        y_val = y_lgb[~train_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Evaluate model
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'train_r2': r2_score(y_train, y_pred_train),
            'val_r2': r2_score(y_val, y_pred_val),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'val_mae': mean_absolute_error(y_val, y_pred_val)
        }
        
        logger.info(f"Model training completed. Validation RMSE: {metrics['val_rmse']:.4f}")
        
        return model, metrics
    
    def train_bayesian_model(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pm.Model, az.InferenceData]:
        """
        Train Bayesian hierarchical model using PyMC
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Tuple of (model, inference_data)
        """
        logger.info("Training Bayesian hierarchical model")
        
        # Prepare data
        feature_cols = [col for col in X.columns if col not in ['mine_id', 'feature_date']]
        X_features = X[feature_cols].select_dtypes(include=[np.number])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        y_target = y['fvi_delta'].values
        
        # Get unique mine IDs for hierarchical structure
        mine_ids = X['mine_id'].unique()
        mine_id_map = {mine_id: i for i, mine_id in enumerate(mine_ids)}
        mine_indices = X['mine_id'].map(mine_id_map).values
        
        n_features = X_scaled.shape[1]
        n_mines = len(mine_ids)
        
        with pm.Model() as model:
            # Hyperpriors
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=1)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=1, shape=n_features)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)
            
            # Mine-specific parameters
            alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_mines)
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=(n_mines, n_features))
            
            # Model error
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear predictor
            linear_pred = alpha[mine_indices] + pm.math.sum(beta[mine_indices] * X_scaled, axis=1)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=linear_pred, sigma=sigma, observed=y_target)
            
            # Sample from posterior
            trace = pm.sample(2000, tune=1000, chains=4, cores=4, target_accept=0.9)
        
        logger.info("Bayesian model training completed")
        
        return model, trace
    
    async def save_model_artifacts(self, model: Any, metrics: Dict[str, float],
                                   model_type: str, feature_names: List[str]) -> str:
        """
        Save model artifacts with MLflow
        
        Args:
            model: Trained model
            metrics: Model metrics
            model_type: Type of model ('lightgbm' or 'bayesian')
            feature_names: List of feature names
            
        Returns:
            Model version string
        """
        with mlflow.start_run() as run:
            # Log parameters
            if model_type == 'lightgbm':
                mlflow.log_params(model.params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            if model_type == 'lightgbm':
                signature = infer_signature(
                    np.random.rand(1, len(feature_names)), 
                    np.array([0.5])
                )
                mlflow.lightgbm.log_model(
                    model, 
                    "model", 
                    signature=signature,
                    input_example=np.random.rand(1, len(feature_names))
                )
            else:  # bayesian
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    pickle.dump(model, f)
                    mlflow.log_artifact(f.name, "model")
                os.unlink(f.name)
            
            # Log feature names
            mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")
            
            # Log scaler
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                joblib.dump(self.scaler, f.name)
                mlflow.log_artifact(f.name, "scaler")
            os.unlink(f.name)
            
            # Log environment
            conda_env = {
                'channels': ['conda-forge'],
                'dependencies': [
                    'python=3.12',
                    'lightgbm',
                    'pymc',
                    'scikit-learn',
                    'numpy',
                    'pandas',
                    'joblib',
                    {
                        'pip': [
                            'mlflow',
                            'optuna',
                            'arviz'
                        ]
                    }
                ]
            }
            
            mlflow.log_dict(conda_env, "conda_env.yaml")
            
            # Tag run
            mlflow.set_tags({
                "model_type": model_type,
                "training_date": datetime.now().isoformat(),
                "feature_count": len(feature_names),
                "data_version": "v1.0"
            })
            
            model_version = run.info.run_id
        
        logger.info(f"Model artifacts saved with version: {model_version}")
        return model_version
    
    async def run_training_pipeline(self, 
                                   start_date: datetime = None,
                                   end_date: datetime = None,
                                   optimize_params: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Training results and model information
        """
        start_time = datetime.now()
        logger.info("Starting complete training pipeline")
        
        try:
            # 1. Hydrate feature store
            data = await self.hydrate_feature_store(start_date, end_date)
            
            if len(data) < 100:
                raise ValueError("Insufficient training data")
            
            # 2. Prepare training data
            X, y = self.prepare_training_data(data)
            
            # 3. Train LightGBM model
            if optimize_params:
                params = self.optimize_lightgbm_params(X, y)
            else:
                params = None
            
            lgb_model, lgb_metrics = await self.train_lightgbm_model(X, y, params)
            
            # 4. Train Bayesian model
            bayesian_model, bayesian_trace = self.train_bayesian_model(X, y)
            
            # 5. Save artifacts
            feature_names = [col for col in X.columns if col not in ['mine_id', 'feature_date']]
            
            lgb_version = await self.save_model_artifacts(
                lgb_model, lgb_metrics, 'lightgbm', feature_names
            )
            
            bayesian_version = await self.save_model_artifacts(
                bayesian_model, {}, 'bayesian', feature_names
            )
            
            # 6. Register models
            await self._register_models(lgb_version, bayesian_version, lgb_metrics)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                "status": "completed",
                "duration_seconds": duration,
                "training_samples": len(X),
                "feature_count": len(feature_names),
                "lgb_model_version": lgb_version,
                "bayesian_model_version": bayesian_version,
                "lgb_metrics": lgb_metrics,
                "timestamp": end_time.isoformat()
            }
            
            logger.info(f"Training pipeline completed in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    async def _register_models(self, lgb_version: str, bayesian_version: str, metrics: Dict[str, float]) -> None:
        """Register models in model registry"""
        
        # Register LightGBM model
        model_name = "fvi_lightgbm_model"
        
        try:
            mlflow.register_model(
                f"runs:/{lgb_version}/model",
                model_name,
                tags={
                    "model_type": "lightgbm",
                    "validation_rmse": metrics["val_rmse"],
                    "validation_r2": metrics["val_r2"]
                }
            )
            
            # Transition to staging if performance is good
            if metrics["val_rmse"] < 0.1 and metrics["val_r2"] > 0.7:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=1,  # Assuming first version
                    stage="Staging"
                )
            
            logger.info(f"Registered LightGBM model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error registering LightGBM model: {str(e)}")
        
        # Store model metadata in database
        await self._store_model_metadata(lgb_version, bayesian_version, metrics)
    
    async def _store_model_metadata(self, lgb_version: str, bayesian_version: str, 
                                   metrics: Dict[str, float]) -> None:
        """Store model metadata in database"""
        
        query = """
        INSERT INTO model_registry (
            model_type, model_version, metrics, created_at, is_active
        ) VALUES ($1, $2, $3, $4, $5)
        """
        
        timestamp = datetime.now()
        
        async with self.db_manager.get_connection() as conn:
            # Store LightGBM model
            await conn.execute(
                query, 
                'lightgbm', 
                lgb_version, 
                json.dumps(metrics), 
                timestamp, 
                True
            )
            
            # Store Bayesian model
            await conn.execute(
                query, 
                'bayesian', 
                bayesian_version, 
                json.dumps({}), 
                timestamp, 
                True
            )
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.feature_store.cleanup()
        await self.weight_engine.cleanup()
        logger.info("ML training pipeline cleanup completed")


# Training scheduler
class TrainingScheduler:
    """Scheduler for automated model training"""
    
    def __init__(self):
        self.pipeline = MLTrainingPipeline()
    
    async def run_scheduled_training(self, schedule_type: str = "daily") -> Dict[str, Any]:
        """Run scheduled training based on schedule type"""
        
        logger.info(f"Starting scheduled training: {schedule_type}")
        
        await self.pipeline.initialize()
        
        try:
            # Determine date range based on schedule
            if schedule_type == "daily":
                start_date = datetime.now() - timedelta(days=30)
                end_date = datetime.now()
            elif schedule_type == "weekly":
                start_date = datetime.now() - timedelta(days=90)
                end_date = datetime.now()
            elif schedule_type == "monthly":
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()
            else:
                raise ValueError(f"Unknown schedule type: {schedule_type}")
            
            # Run training pipeline
            results = await self.pipeline.run_training_pipeline(
                start_date=start_date,
                end_date=end_date,
                optimize_params=(schedule_type in ["weekly", "monthly"])
            )
            
            return results
            
        finally:
            await self.pipeline.cleanup()


if __name__ == "__main__":
    # Example usage
    async def main():
        scheduler = TrainingScheduler()
        results = await scheduler.run_scheduled_training("daily")
        print(json.dumps(results, indent=2))
    
    asyncio.run(main())
