import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import shap
import joblib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..config import settings

logger = logging.getLogger(__name__)


class FVIPredictor:
    """Main FVI prediction model using ensemble methods."""
    
    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = "fvi_score"
        self.is_fitted = False
        self.feature_importance = {}
        self.shap_explainer = None
        self.model_metadata = {}
        
    def prepare_data(self, data: pd.DataFrame, target_col: str = "fvi_score") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        # Separate features and target
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the FVI prediction model."""
        logger.info(f"Training {self.model_type} model with {len(X)} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features for some models
        if self.model_type in ["linear", "neural"]:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Initialize model based on type
        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
        elif self.model_type == "catboost":
            self.model = cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                loss_function='RMSE',
                random_seed=42,
                verbose=False
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        start_time = datetime.now()
        
        if self.model_type == "lightgbm":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
        elif self.model_type == "catboost":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        elif self.model_type == "xgboost":
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_val = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importance))
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        # Initialize SHAP explainer
        try:
            if self.model_type in ["lightgbm", "xgboost", "catboost"]:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.LinearExplainer(self.model, X_train_scaled)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
        
        self.is_fitted = True
        
        # Store metadata
        self.model_metadata = {
            "model_type": self.model_type,
            "training_time": training_time,
            "feature_count": len(self.feature_names),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "trained_at": datetime.now().isoformat()
        }
        
        return {
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "cv_rmse_mean": cv_rmse.mean(),
            "cv_rmse_std": cv_rmse.std(),
            "feature_importance": self.feature_importance,
            "training_time": training_time,
            "metadata": self.model_metadata
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure same feature order
        X = X[self.feature_names]
        
        # Handle categorical variables and missing values
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.fillna(X.mean())
        
        # Scale if needed
        if self.model_type in ["linear", "neural"]:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        predictions = self.predict(X)
        
        # Simple confidence interval based on model type
        if self.model_type == "random_forest":
            # Use prediction std from trees
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            confidence = np.std(tree_predictions, axis=0)
        else:
            # Use cross-validation RMSE as confidence proxy
            cv_rmse = self.model_metadata.get("cv_rmse_mean", 5.0)
            confidence = np.full_like(predictions, cv_rmse)
        
        return predictions, confidence
    
    def explain_prediction(self, X: pd.DataFrame, max_features: int = 10) -> Dict[str, Any]:
        """Explain predictions using SHAP."""
        if not self.shap_explainer:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Get SHAP values
            if self.model_type in ["linear", "neural"]:
                X_scaled = self.scaler.transform(X)
                shap_values = self.shap_explainer.shap_values(X_scaled)
            else:
                shap_values = self.shap_explainer.shap_values(X)
            
            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(self.feature_names):
                contribution = np.mean(shap_values[:, i]) if shap_values.ndim > 1 else shap_values[i]
                feature_contributions[feature] = float(contribution)
            
            # Sort by absolute contribution
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:max_features]
            
            return {
                "feature_contributions": dict(sorted_features),
                "base_value": float(self.shap_explainer.expected_value),
                "prediction": float(self.predict(X)[0]),
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
            }
        
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {"error": str(e)}
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance,
            "metadata": self.model_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.target_name = model_data["target_name"]
            self.model_type = model_data["model_type"]
            self.feature_importance = model_data["feature_importance"]
            self.model_metadata = model_data["metadata"]
            self.is_fitted = True
            
            # Reinitialize SHAP explainer
            try:
                if self.model_type in ["lightgbm", "xgboost", "catboost"]:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # Need training data for linear explainer - skip for now
                    pass
            except Exception as e:
                logger.warning(f"Could not reinitialize SHAP explainer: {e}")
            
            logger.info(f"Model loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning")
        
        if self.model_type == "lightgbm":
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'min_child_samples': [10, 20, 30]
            }
            base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        elif self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42)
        
        else:
            logger.warning(f"Hyperparameter tuning not implemented for {self.model_type}")
            return {"error": "Hyperparameter tuning not implemented"}
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": -grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }


class EnsemblePredictor:
    """Ensemble of multiple FVI predictors."""
    
    def __init__(self, model_types: List[str] = None):
        if model_types is None:
            model_types = ["lightgbm", "catboost", "random_forest"]
        
        self.model_types = model_types
        self.models = {model_type: FVIPredictor(model_type) for model_type in model_types}
        self.weights = None
        self.is_fitted = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        results = {}
        predictions = {}
        
        # Train each model
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model")
            model_results = model.train(X, y)
            results[model_type] = model_results
            
            # Get validation predictions for weight calculation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            val_predictions = model.predict(X_val)
            predictions[model_type] = val_predictions
        
        # Calculate weights based on validation performance
        weights = {}
        for model_type in self.model_types:
            rmse = results[model_type]["val_rmse"]
            # Weight is inversely proportional to RMSE
            weights[model_type] = 1 / (rmse + 1e-6)
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v / total_weight for k, v in weights.items()}
        
        self.is_fitted = True
        
        return {
            "individual_results": results,
            "ensemble_weights": self.weights,
            "ensemble_val_rmse": self._calculate_ensemble_rmse(predictions, y_val)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from all models
        predictions = {}
        for model_type, model in self.models.items():
            predictions[model_type] = model.predict(X)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for model_type, weight in self.weights.items():
            ensemble_pred += weight * predictions[model_type]
        
        return ensemble_pred
    
    def _calculate_ensemble_rmse(self, predictions: Dict[str, np.ndarray], y_true: pd.Series) -> float:
        """Calculate ensemble RMSE."""
        ensemble_pred = np.zeros(len(y_true))
        for model_type, weight in self.weights.items():
            ensemble_pred += weight * predictions[model_type]
        
        return np.sqrt(mean_squared_error(y_true, ensemble_pred))


# Export classes
__all__ = ["FVIPredictor", "EnsemblePredictor"]
