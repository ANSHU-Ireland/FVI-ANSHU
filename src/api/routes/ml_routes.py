from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib
import os

from ...database import get_db, MetricValueCRUD, CompositeScoreCRUD, ModelRunCRUD
from ...models import (
    FVIAnalysisRequest, FVIAnalysisResponse,
    ScenarioAnalysisRequest, ScenarioAnalysisResponse,
    ModelRunCreate, ModelRunResponse,
    CompositeScoreCreate
)
from ...ml import FVIPredictor, EnsemblePredictor, HierarchicalBayesianModel, BayesianScenarioAnalysis
from ...data import MockDataGenerator
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Global model cache
MODEL_CACHE = {}


def load_model(model_type: str = "lightgbm") -> FVIPredictor:
    """Load or create FVI model."""
    cache_key = f"fvi_model_{model_type}"
    
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    # Try to load existing model
    model_path = f"{settings.MODELS_DIR}/fvi_model_{model_type}.pkl"
    
    if os.path.exists(model_path):
        try:
            model = FVIPredictor(model_type)
            model.load_model(model_path)
            MODEL_CACHE[cache_key] = model
            logger.info(f"Loaded existing {model_type} model")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    # Create new model
    model = FVIPredictor(model_type)
    MODEL_CACHE[cache_key] = model
    logger.info(f"Created new {model_type} model")
    return model


@router.post("/train-model", response_model=Dict[str, Any])
async def train_model(
    model_type: str = "lightgbm",
    sub_industry: str = "Coal Mining",
    country: str = "Global",
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Train FVI prediction model."""
    try:
        # Generate or load training data
        logger.info(f"Training {model_type} model for {sub_industry}")
        
        # For demo, use mock data
        mock_data = MockDataGenerator.generate_coal_metrics_data(
            sub_industry, country, list(range(2015, 2024))
        )
        
        # Convert to DataFrame
        training_rows = []
        for year_idx, year in enumerate(mock_data["years"]):
            row = {"year": year, "sub_industry": sub_industry, "country": country}
            for metric_name, values in mock_data["metrics"].items():
                row[metric_name] = values[year_idx]
            row["fvi_score"] = mock_data["fvi_scores"][year_idx]
            training_rows.append(row)
        
        df = pd.DataFrame(training_rows)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ["fvi_score", "year", "sub_industry", "country"]]
        X = df[feature_cols]
        y = df["fvi_score"]
        
        # Train model
        model = load_model(model_type)
        training_results = model.train(X, y)
        
        # Save model
        model_path = f"{settings.MODELS_DIR}/fvi_model_{model_type}.pkl"
        model.save_model(model_path)
        
        # Create model run record
        model_run = ModelRunCreate(
            model_name=f"FVI_{model_type}",
            model_version=f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            training_start=datetime.now(),
            training_end=datetime.now(),
            training_duration_seconds=int(training_results["training_time"]),
            rmse=training_results["val_rmse"],
            r2_score=training_results["val_r2"],
            mae=training_results["val_mae"],
            cross_validation_scores={"cv_rmse_mean": training_results["cv_rmse_mean"]},
            hyperparameters={"model_type": model_type},
            feature_importance=training_results["feature_importance"],
            model_file_path=model_path,
            data_version="mock_v1",
            is_deployed=True,
            deployment_date=datetime.now()
        )
        
        db_model_run = ModelRunCRUD.create(db, model_run)
        
        return {
            "message": "Model training completed successfully",
            "model_type": model_type,
            "model_run_id": db_model_run.id,
            "training_results": training_results,
            "model_path": model_path,
            "feature_importance": training_results["feature_importance"]
        }
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=FVIAnalysisResponse)
async def predict_fvi(
    request: FVIAnalysisRequest,
    model_type: str = "lightgbm",
    db: Session = Depends(get_db)
):
    """Predict FVI score for given industry and parameters."""
    try:
        # Load model
        model = load_model(model_type)
        
        if not model.is_fitted:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first."
            )
        
        # Get or generate data for prediction
        mock_data = MockDataGenerator.generate_coal_metrics_data(
            request.sub_industry, request.country or "Global", [2023]
        )
        
        # Create prediction DataFrame
        prediction_row = {}
        for metric_name, values in mock_data["metrics"].items():
            prediction_row[metric_name] = values[0]  # Use first (and only) year
        
        # Apply scenario changes if provided
        if request.scenario:
            for metric_name, change in request.scenario.items():
                if metric_name in prediction_row:
                    prediction_row[metric_name] += change
        
        pred_df = pd.DataFrame([prediction_row])
        
        # Make prediction
        prediction = model.predict(pred_df)[0]
        
        # Get prediction confidence
        try:
            prediction_with_conf = model.predict_with_confidence(pred_df)
            confidence_score = 1.0 - (prediction_with_conf[1][0] / 100.0)  # Convert to 0-1 scale
        except:
            confidence_score = 0.8  # Default confidence
        
        # Get SHAP explanations
        try:
            shap_explanation = model.explain_prediction(pred_df)
            shap_values = shap_explanation.get("feature_contributions", {})
        except:
            shap_values = {}
        
        # Calculate percentile (simplified)
        percentile = min(100, max(0, (prediction / 100) * 100))
        
        # Generate component scores
        component_scores = {}
        for metric_name, value in prediction_row.items():
            component_scores[metric_name] = value
        
        # Generate insights and recommendations
        key_insights = [
            f"FVI score of {prediction:.1f} indicates {'high' if prediction > 60 else 'moderate' if prediction > 40 else 'low'} future viability",
            f"Strongest contributing factors: {', '.join(list(shap_values.keys())[:3])}" if shap_values else "Model explanation not available",
            f"Confidence level: {confidence_score:.1%}"
        ]
        
        recommendations = [
            "Focus on improving lowest-scoring metrics",
            "Monitor regulatory changes affecting emissions and ecological scores",
            "Consider diversification strategies for long-term viability"
        ]
        
        return FVIAnalysisResponse(
            sub_industry=request.sub_industry,
            country=request.country,
            region=request.region,
            horizon=request.horizon,
            fvi_score=prediction,
            fvi_percentile=percentile,
            component_scores=component_scores,
            shap_explanations=shap_values,
            confidence_score=confidence_score,
            key_insights=key_insights,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error making FVI prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario-analysis", response_model=ScenarioAnalysisResponse)
async def scenario_analysis(
    request: ScenarioAnalysisRequest,
    model_type: str = "lightgbm",
    db: Session = Depends(get_db)
):
    """Perform scenario analysis on FVI prediction."""
    try:
        # Get baseline prediction
        baseline_response = await predict_fvi(request.base_analysis, model_type, db)
        baseline_score = baseline_response.fvi_score
        
        # Get scenario prediction
        scenario_request = request.base_analysis.copy()
        scenario_request.scenario = request.scenario_changes
        scenario_response = await predict_fvi(scenario_request, model_type, db)
        scenario_score = scenario_response.fvi_score
        
        # Calculate changes
        score_change = scenario_score - baseline_score
        percentage_change = (score_change / baseline_score) * 100 if baseline_score != 0 else 0
        
        # Analyze affected metrics
        affected_metrics = {}
        for metric_name, change in request.scenario_changes.items():
            if metric_name in baseline_response.component_scores:
                old_value = baseline_response.component_scores[metric_name]
                new_value = scenario_response.component_scores[metric_name]
                impact = new_value - old_value
                
                affected_metrics[metric_name] = {
                    "old_value": old_value,
                    "new_value": new_value,
                    "impact": impact
                }
        
        # Generate explanation
        direction = "increase" if score_change > 0 else "decrease"
        magnitude = "significant" if abs(percentage_change) > 5 else "moderate" if abs(percentage_change) > 2 else "minor"
        
        explanation = f"The scenario results in a {magnitude} {direction} in FVI score by {abs(score_change):.1f} points ({abs(percentage_change):.1f}%). "
        explanation += f"Key drivers: {', '.join(list(affected_metrics.keys())[:3])}"
        
        return ScenarioAnalysisResponse(
            base_fvi_score=baseline_score,
            scenario_fvi_score=scenario_score,
            score_change=score_change,
            percentage_change=percentage_change,
            affected_metrics=affected_metrics,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Error performing scenario analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-performance", response_model=Dict[str, Any])
async def get_model_performance(
    model_type: str = "lightgbm",
    db: Session = Depends(get_db)
):
    """Get model performance metrics."""
    try:
        # Get latest model run
        model_run = ModelRunCRUD.get_latest_by_model_name(db, f"FVI_{model_type}")
        
        if not model_run:
            raise HTTPException(status_code=404, detail="No model runs found")
        
        # Get model from cache
        model = MODEL_CACHE.get(f"fvi_model_{model_type}")
        
        performance_metrics = {
            "model_type": model_type,
            "model_run_id": model_run.id,
            "training_time": model_run.training_duration_seconds,
            "rmse": model_run.rmse,
            "r2_score": model_run.r2_score,
            "mae": model_run.mae,
            "is_deployed": model_run.is_deployed,
            "deployment_date": model_run.deployment_date.isoformat() if model_run.deployment_date else None,
            "feature_importance": model_run.feature_importance,
            "cross_validation_scores": model_run.cross_validation_scores
        }
        
        if model:
            performance_metrics["model_metadata"] = model.model_metadata
        
        return performance_metrics
    
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-ensemble", response_model=Dict[str, Any])
async def train_ensemble_model(
    model_types: List[str] = ["lightgbm", "catboost", "random_forest"],
    sub_industry: str = "Coal Mining",
    country: str = "Global",
    db: Session = Depends(get_db)
):
    """Train ensemble of FVI models."""
    try:
        logger.info(f"Training ensemble with models: {model_types}")
        
        # Generate training data
        mock_data = MockDataGenerator.generate_coal_metrics_data(
            sub_industry, country, list(range(2015, 2024))
        )
        
        # Convert to DataFrame
        training_rows = []
        for year_idx, year in enumerate(mock_data["years"]):
            row = {"year": year, "sub_industry": sub_industry, "country": country}
            for metric_name, values in mock_data["metrics"].items():
                row[metric_name] = values[year_idx]
            row["fvi_score"] = mock_data["fvi_scores"][year_idx]
            training_rows.append(row)
        
        df = pd.DataFrame(training_rows)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ["fvi_score", "year", "sub_industry", "country"]]
        X = df[feature_cols]
        y = df["fvi_score"]
        
        # Train ensemble
        ensemble = EnsemblePredictor(model_types)
        ensemble_results = ensemble.train(X, y)
        
        # Save ensemble
        ensemble_path = f"{settings.MODELS_DIR}/fvi_ensemble.pkl"
        joblib.dump(ensemble, ensemble_path)
        
        # Cache ensemble
        MODEL_CACHE["fvi_ensemble"] = ensemble
        
        return {
            "message": "Ensemble training completed successfully",
            "model_types": model_types,
            "ensemble_results": ensemble_results,
            "ensemble_path": ensemble_path,
            "weights": ensemble_results["ensemble_weights"]
        }
    
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperparameter-tuning", response_model=Dict[str, Any])
async def hyperparameter_tuning(
    model_type: str = "lightgbm",
    sub_industry: str = "Coal Mining",
    country: str = "Global",
    db: Session = Depends(get_db)
):
    """Perform hyperparameter tuning for FVI model."""
    try:
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        # Generate training data
        mock_data = MockDataGenerator.generate_coal_metrics_data(
            sub_industry, country, list(range(2015, 2024))
        )
        
        # Convert to DataFrame
        training_rows = []
        for year_idx, year in enumerate(mock_data["years"]):
            row = {"year": year, "sub_industry": sub_industry, "country": country}
            for metric_name, values in mock_data["metrics"].items():
                row[metric_name] = values[year_idx]
            row["fvi_score"] = mock_data["fvi_scores"][year_idx]
            training_rows.append(row)
        
        df = pd.DataFrame(training_rows)
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ["fvi_score", "year", "sub_industry", "country"]]
        X = df[feature_cols]
        y = df["fvi_score"]
        
        # Perform hyperparameter tuning
        model = FVIPredictor(model_type)
        tuning_results = model.hyperparameter_tuning(X, y)
        
        # Train with best parameters
        training_results = model.train(X, y)
        
        # Save tuned model
        model_path = f"{settings.MODELS_DIR}/fvi_model_{model_type}_tuned.pkl"
        model.save_model(model_path)
        
        # Update cache
        MODEL_CACHE[f"fvi_model_{model_type}"] = model
        
        return {
            "message": "Hyperparameter tuning completed successfully",
            "model_type": model_type,
            "tuning_results": tuning_results,
            "training_results": training_results,
            "model_path": model_path
        }
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance", response_model=Dict[str, Any])
async def get_feature_importance(
    model_type: str = "lightgbm",
    top_n: int = 10
):
    """Get feature importance from trained model."""
    try:
        model = MODEL_CACHE.get(f"fvi_model_{model_type}")
        
        if not model or not model.is_fitted:
            raise HTTPException(
                status_code=400, 
                detail="Model not found or not trained"
            )
        
        # Get feature importance
        feature_importance = model.feature_importance
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return {
            "model_type": model_type,
            "top_features": dict(sorted_features),
            "total_features": len(feature_importance),
            "feature_names": model.feature_names
        }
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-predict", response_model=Dict[str, Any])
async def batch_predict(
    sub_industries: List[str],
    countries: List[str] = ["Global"],
    horizon: int = 10,
    model_type: str = "lightgbm",
    db: Session = Depends(get_db)
):
    """Perform batch FVI predictions."""
    try:
        model = load_model(model_type)
        
        if not model.is_fitted:
            raise HTTPException(
                status_code=400, 
                detail="Model not trained. Please train the model first."
            )
        
        batch_results = []
        
        for sub_industry in sub_industries:
            for country in countries:
                # Create prediction request
                request = FVIAnalysisRequest(
                    sub_industry=sub_industry,
                    country=country,
                    horizon=horizon
                )
                
                # Make prediction
                prediction = await predict_fvi(request, model_type, db)
                
                batch_results.append({
                    "sub_industry": sub_industry,
                    "country": country,
                    "horizon": horizon,
                    "fvi_score": prediction.fvi_score,
                    "fvi_percentile": prediction.fvi_percentile,
                    "confidence_score": prediction.confidence_score
                })
        
        return {
            "message": "Batch prediction completed successfully",
            "total_predictions": len(batch_results),
            "results": batch_results,
            "model_type": model_type
        }
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
