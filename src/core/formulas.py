"""
FVI Formula Engine
==================

Implements the mathematical formulas for each FVI metric sheet
with support for vectorized operations and dynamic evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union
import logging
import math
from scipy.stats import zscore
from scipy.special import expit  # sigmoid function

from .naming import FVIMetricFormulas, FVINamingConventions

logger = logging.getLogger(__name__)


class FormulaEngine:
    """Engine for evaluating FVI metric formulas."""
    
    def __init__(self):
        self.custom_functions = {
            'log1p': np.log1p,
            'zscore': self._safe_zscore,
            'sigmoid': expit,
            'avg': np.mean,
            'max': np.max,
            'min': np.min,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs
        }
        
    def _safe_zscore(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Safe z-score calculation that handles edge cases."""
        x = np.asarray(x)
        if len(x) <= 1:
            return np.zeros_like(x)
        
        std = np.std(x, ddof=1)
        if std == 0:
            return np.zeros_like(x)
        
        return (x - np.mean(x)) / std
    
    def evaluate_formula(self, formula: str, data: Dict[str, Any]) -> Union[float, np.ndarray, List[float]]:
        """
        Evaluate a formula with given data.
        
        Args:
            formula: Mathematical formula string
            data: Dictionary of variable names and values
            
        Returns:
            Result of formula evaluation
        """
        try:
            # Create safe evaluation environment
            safe_dict = {
                '__builtins__': {},
                **self.custom_functions,
                **data
            }
            
            # Replace any remaining function calls
            processed_formula = self._preprocess_formula(formula)
            
            # Evaluate the formula
            result = eval(processed_formula, safe_dict)
            
            # Ensure result is numeric
            if isinstance(result, (list, np.ndarray)):
                return np.asarray(result, dtype=float)
            else:
                return float(result)
                
        except Exception as e:
            logger.error(f"Error evaluating formula '{formula}': {e}")
            logger.error(f"Available data keys: {list(data.keys())}")
            return np.nan
    
    def _preprocess_formula(self, formula: str) -> str:
        """Preprocess formula to handle special cases."""
        # Handle division by zero
        formula = formula.replace('/', ' / ')
        
        # Handle percentage calculations
        formula = formula.replace('_pct', '_pct')
        
        # Handle special mathematical functions
        replacements = {
            'avg(': 'np.mean(',
            'max(': 'np.max(',
            'min(': 'np.min(',
        }
        
        for old, new in replacements.items():
            formula = formula.replace(old, new)
        
        return formula
    
    def calculate_sheet_metrics(self, sheet_number: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all metrics for a specific sheet.
        
        Args:
            sheet_number: Sheet number (e.g., "01", "04")
            data: DataFrame with raw data
            
        Returns:
            DataFrame with calculated feature columns
        """
        formulas = FVIMetricFormulas.get_sheet_formulas(sheet_number)
        results = data.copy()
        
        for metric_slug, formula in formulas.items():
            try:
                # Generate standardized column names
                metric_key = f"{sheet_number}_{metric_slug}"
                feature_col = FVINamingConventions.generate_feature_column(metric_key)
                
                # Calculate metric for each row
                metric_values = []
                for _, row in data.iterrows():
                    row_data = row.to_dict()
                    value = self.evaluate_formula(formula, row_data)
                    metric_values.append(value)
                
                results[feature_col] = metric_values
                logger.info(f"Calculated {feature_col} using formula: {formula}")
                
            except Exception as e:
                logger.error(f"Error calculating metric {metric_slug}: {e}")
                # Fill with NaN if calculation fails
                feature_col = FVINamingConventions.generate_feature_column(f"{sheet_number}_{metric_slug}")
                results[feature_col] = np.nan
        
        return results
    
    def calculate_necessity_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 1 - Necessity Score metrics."""
        return self.calculate_sheet_metrics("01", data)
    
    def calculate_resource_scarcity_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 2 - Resource Extraction & Scarcity metrics."""
        return self.calculate_sheet_metrics("02", data)
    
    def calculate_artificial_support_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 3 - Artificial Support metrics."""
        return self.calculate_sheet_metrics("03", data)
    
    def calculate_emissions_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 4 - Emissions metrics."""
        return self.calculate_sheet_metrics("04", data)
    
    def calculate_ecological_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 5 - Ecological metrics."""
        return self.calculate_sheet_metrics("05", data)
    
    def calculate_workforce_transition_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 8 - Workforce Transition metrics."""
        return self.calculate_sheet_metrics("08", data)
    
    def calculate_infrastructure_repurposing_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 9 - Infrastructure Repurposing metrics."""
        return self.calculate_sheet_metrics("09", data)
    
    def calculate_monopoly_control_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 11 - Monopoly & Corporate Control metrics."""
        return self.calculate_sheet_metrics("11", data)
    
    def calculate_economic_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 20 - Economic metrics."""
        return self.calculate_sheet_metrics("20", data)
    
    def calculate_technological_disruption_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sheet 24 - Technological Disruption metrics."""
        return self.calculate_sheet_metrics("24", data)
    
    def calculate_all_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all FVI metrics."""
        sheet_methods = [
            ("01", self.calculate_necessity_metrics),
            ("02", self.calculate_resource_scarcity_metrics),
            ("03", self.calculate_artificial_support_metrics),
            ("04", self.calculate_emissions_metrics),
            ("05", self.calculate_ecological_metrics),
            ("08", self.calculate_workforce_transition_metrics),
            ("09", self.calculate_infrastructure_repurposing_metrics),
            ("11", self.calculate_monopoly_control_metrics),
            ("20", self.calculate_economic_metrics),
            ("24", self.calculate_technological_disruption_metrics)
        ]
        
        results = data.copy()
        
        for sheet_number, method in sheet_methods:
            try:
                sheet_results = method(data)
                # Merge feature columns
                feature_cols = [col for col in sheet_results.columns if col.startswith(f"f_{sheet_number}_")]
                for col in feature_cols:
                    results[col] = sheet_results[col]
                logger.info(f"Calculated {len(feature_cols)} metrics for sheet {sheet_number}")
            except Exception as e:
                logger.error(f"Error calculating metrics for sheet {sheet_number}: {e}")
        
        return results
    
    def validate_formula(self, formula: str, sample_data: Dict[str, Any]) -> bool:
        """Validate if a formula can be evaluated with sample data."""
        try:
            result = self.evaluate_formula(formula, sample_data)
            return not (np.isnan(result) if isinstance(result, (int, float)) else np.isnan(result).all())
        except Exception:
            return False
    
    def get_formula_variables(self, formula: str) -> List[str]:
        """Extract variable names from a formula."""
        import re
        
        # Find all variable names (letters, numbers, underscores)
        variables = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula)
        
        # Remove function names and Python keywords
        excluded = set(self.custom_functions.keys()) | {'and', 'or', 'not', 'in', 'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from', 'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'lambda', 'global', 'nonlocal', 'True', 'False', 'None'}
        
        return [var for var in set(variables) if var not in excluded]


class DynamicWeightingEngine:
    """Engine for dynamic weighting based on data quality and information gain."""
    
    def __init__(self):
        self.base_weights = {}
        self.quality_weights = {}
        self.info_gain_weights = {}
        
    def set_base_weights(self, weights: Dict[str, float]):
        """Set base weights from Excel sheet."""
        self.base_weights = weights.copy()
        
    def calculate_data_quality_weights(self, metrics_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality weights for each metric."""
        quality_weights = {}
        
        for col in metrics_data.columns:
            if col.startswith('f_'):
                # Calculate data quality components
                values = metrics_data[col].dropna()
                
                if len(values) == 0:
                    quality_weights[col] = 0.0
                    continue
                
                # Freshness (assume more recent data is better)
                freshness = 1.0  # Placeholder - would be calculated from data timestamps
                
                # Coverage (fraction of non-null values)
                coverage = len(values) / len(metrics_data)
                
                # Confidence (inverse of coefficient of variation)
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                confidence = 1.0 / (1.0 + cv)  # Normalize to 0-1
                
                # Calculate composite quality score
                from .naming import FVIDataQualityWeights
                quality_score = FVIDataQualityWeights.calculate_dq_score(
                    freshness, coverage, confidence
                )
                
                quality_weights[col] = quality_score
        
        self.quality_weights = quality_weights
        return quality_weights
    
    def calculate_information_gain_weights(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate information gain weights for each feature."""
        from sklearn.feature_selection import mutual_info_regression
        
        info_gain_weights = {}
        
        try:
            # Get feature columns
            feature_cols = [col for col in features.columns if col.startswith('f_')]
            
            if len(feature_cols) == 0:
                return info_gain_weights
            
            # Prepare data
            X = features[feature_cols].fillna(0)
            y = target.fillna(target.mean())
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Normalize to 0-1
            max_mi = np.max(mi_scores) if np.max(mi_scores) > 0 else 1.0
            
            for i, col in enumerate(feature_cols):
                info_gain_weights[col] = mi_scores[i] / max_mi
            
        except Exception as e:
            logger.error(f"Error calculating information gain weights: {e}")
            # Fallback to uniform weights
            for col in feature_cols:
                info_gain_weights[col] = 1.0
        
        self.info_gain_weights = info_gain_weights
        return info_gain_weights
    
    def calculate_dynamic_weights(self, 
                                metrics_data: pd.DataFrame, 
                                target: pd.Series, 
                                horizon: str = "H10") -> Dict[str, float]:
        """Calculate dynamic weights combining quality and information gain."""
        
        # Calculate component weights
        quality_weights = self.calculate_data_quality_weights(metrics_data)
        info_gain_weights = self.calculate_information_gain_weights(metrics_data, target)
        
        # Combine weights
        dynamic_weights = {}
        
        for col in metrics_data.columns:
            if col.startswith('f_'):
                # Get base weight
                base_weight = self.base_weights.get(col, 1.0)
                
                # Get quality weight
                quality_weight = quality_weights.get(col, 1.0)
                
                # Get information gain weight
                info_gain_weight = info_gain_weights.get(col, 1.0)
                
                # Combine weights (weighted average)
                combined_weight = base_weight * (0.5 * quality_weight + 0.5 * info_gain_weight)
                
                # Generate weight column name
                metric_key = col.replace('f_', '')
                from .naming import FVIHorizon
                horizon_enum = FVIHorizon(horizon)
                weight_col = FVINamingConventions.generate_weight_column(metric_key, horizon_enum)
                
                dynamic_weights[weight_col] = combined_weight
        
        return dynamic_weights
    
    def update_weights_with_user_overrides(self, 
                                         current_weights: Dict[str, float], 
                                         user_overrides: Dict[str, float]) -> Dict[str, float]:
        """Update weights with user-provided overrides."""
        updated_weights = current_weights.copy()
        
        for weight_name, override_value in user_overrides.items():
            if weight_name in updated_weights:
                updated_weights[weight_name] = override_value
                logger.info(f"Updated weight {weight_name} to {override_value}")
        
        return updated_weights
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return weights
        
        return {k: v / total_weight for k, v in weights.items()}


# Export classes
__all__ = [
    "FormulaEngine",
    "DynamicWeightingEngine"
]
