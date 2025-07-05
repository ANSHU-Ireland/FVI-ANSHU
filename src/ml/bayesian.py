import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import pymc as pm
import arviz as az
import theano.tensor as tt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class HierarchicalBayesianModel:
    """Hierarchical Bayesian model for FVI prediction with uncertainty quantification."""
    
    def __init__(self, industry_groups: List[str] = None):
        self.industry_groups = industry_groups or []
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.group_encoder = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, group_col: str = "sub_industry") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for Bayesian modeling."""
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != group_col:
                # Use label encoding for other categorical variables
                unique_vals = X[col].unique()
                encoder = {val: i for i, val in enumerate(unique_vals)}
                X[col] = X[col].map(encoder)
        
        # Handle group variable
        if group_col in X.columns:
            unique_groups = X[group_col].unique()
            self.group_encoder = {group: i for i, group in enumerate(unique_groups)}
            group_idx = X[group_col].map(self.group_encoder).values
            X = X.drop(columns=[group_col])
        else:
            # Create dummy group if no group column
            group_idx = np.zeros(len(X), dtype=int)
            self.group_encoder = {"default": 0}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        
        return X_scaled, y.values, group_idx
    
    def build_model(self, X: np.ndarray, y: np.ndarray, group_idx: np.ndarray) -> pm.Model:
        """Build hierarchical Bayesian model."""
        n_features = X.shape[1]
        n_groups = len(np.unique(group_idx))
        
        with pm.Model() as model:
            # Hyperpriors for group-level parameters
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
            sigma_alpha = pm.HalfCauchy("sigma_alpha", beta=2.5)
            
            mu_beta = pm.Normal("mu_beta", mu=0, sigma=10, shape=n_features)
            sigma_beta = pm.HalfCauchy("sigma_beta", beta=2.5, shape=n_features)
            
            # Group-level parameters
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
            beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=(n_groups, n_features))
            
            # Model error
            sigma_y = pm.HalfCauchy("sigma_y", beta=2.5)
            
            # Linear model
            mu = alpha[group_idx] + pm.math.dot(X, beta[group_idx].T).diagonal()
            
            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y)
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, group_col: str = "sub_industry", 
              n_samples: int = 2000, n_tune: int = 1000, chains: int = 4) -> Dict[str, Any]:
        """Train the hierarchical Bayesian model."""
        logger.info("Training hierarchical Bayesian model")
        
        # Prepare data
        X_scaled, y_array, group_idx = self.prepare_data(X, y, group_col)
        
        # Build model
        self.model = self.build_model(X_scaled, y_array, group_idx)
        
        # Sample from posterior
        with self.model:
            start_time = datetime.now()
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=chains,
                cores=4,
                return_inferencedata=True,
                random_seed=42
            )
            training_time = (datetime.now() - start_time).total_seconds()
        
        self.is_fitted = True
        
        # Calculate model diagnostics
        diagnostics = self._calculate_diagnostics()
        
        return {
            "training_time": training_time,
            "n_samples": n_samples,
            "n_tune": n_tune,
            "chains": chains,
            "diagnostics": diagnostics,
            "model_summary": str(az.summary(self.trace))
        }
    
    def predict(self, X: pd.DataFrame, group_col: str = "sub_industry", 
                n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled, _, group_idx = self.prepare_data(X, pd.Series([0] * len(X)), group_col)
        
        # Sample from posterior predictive distribution
        with self.model:
            # Update model with new data
            pm.set_data({"X": X_scaled, "group_idx": group_idx})
            
            # Sample from posterior predictive
            posterior_pred = pm.sample_posterior_predictive(
                self.trace, samples=n_samples, random_seed=42
            )
        
        # Extract predictions
        predictions = posterior_pred.posterior_predictive["y_obs"].values
        
        # Calculate mean and uncertainty
        pred_mean = np.mean(predictions, axis=(0, 1))
        pred_std = np.std(predictions, axis=(0, 1))
        
        return pred_mean, pred_std
    
    def predict_with_intervals(self, X: pd.DataFrame, group_col: str = "sub_industry",
                              credible_interval: float = 0.95) -> Dict[str, np.ndarray]:
        """Make predictions with credible intervals."""
        pred_mean, pred_std = self.predict(X, group_col)
        
        # Calculate credible intervals
        alpha = 1 - credible_interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Use normal approximation for intervals
        from scipy.stats import norm
        lower_bound = norm.ppf(lower_percentile / 100, pred_mean, pred_std)
        upper_bound = norm.ppf(upper_percentile / 100, pred_mean, pred_std)
        
        return {
            "mean": pred_mean,
            "std": pred_std,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "credible_interval": credible_interval
        }
    
    def _calculate_diagnostics(self) -> Dict[str, Any]:
        """Calculate model diagnostics."""
        try:
            # R-hat diagnostic
            rhat = az.rhat(self.trace)
            
            # Effective sample size
            ess = az.ess(self.trace)
            
            # Monte Carlo standard error
            mcse = az.mcse(self.trace)
            
            return {
                "rhat_max": float(rhat.max()),
                "rhat_mean": float(rhat.mean()),
                "ess_min": float(ess.min()),
                "ess_mean": float(ess.mean()),
                "mcse_mean": float(mcse.mean()),
                "n_divergences": int(self.trace.sample_stats.diverging.sum()),
                "accept_rate": float(self.trace.sample_stats.accept.mean())
            }
        except Exception as e:
            logger.error(f"Error calculating diagnostics: {e}")
            return {"error": str(e)}
    
    def get_group_effects(self) -> Dict[str, Any]:
        """Get group-level effects from the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting group effects")
        
        try:
            # Extract group-level parameters
            alpha_samples = self.trace.posterior["alpha"].values
            beta_samples = self.trace.posterior["beta"].values
            
            # Calculate summary statistics
            alpha_mean = np.mean(alpha_samples, axis=(0, 1))
            alpha_std = np.std(alpha_samples, axis=(0, 1))
            
            beta_mean = np.mean(beta_samples, axis=(0, 1))
            beta_std = np.std(beta_samples, axis=(0, 1))
            
            # Map back to group names
            group_names = list(self.group_encoder.keys())
            
            group_effects = {}
            for i, group_name in enumerate(group_names):
                group_effects[group_name] = {
                    "intercept_mean": float(alpha_mean[i]),
                    "intercept_std": float(alpha_std[i]),
                    "coefficients_mean": beta_mean[i].tolist(),
                    "coefficients_std": beta_std[i].tolist(),
                    "feature_names": self.feature_names
                }
            
            return group_effects
            
        except Exception as e:
            logger.error(f"Error extracting group effects: {e}")
            return {"error": str(e)}
    
    def plot_diagnostics(self, save_path: str = None):
        """Plot model diagnostics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        
        try:
            import matplotlib.pyplot as plt
            
            # Create diagnostic plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Trace plots
            az.plot_trace(self.trace, var_names=["mu_alpha", "sigma_alpha"], ax=axes[0])
            
            # R-hat plot
            az.plot_rhat(self.trace, ax=axes[1, 0])
            
            # Effective sample size plot
            az.plot_ess(self.trace, ax=axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {e}")
            return None


class BayesianScenarioAnalysis:
    """Bayesian scenario analysis for FVI predictions."""
    
    def __init__(self, base_model: HierarchicalBayesianModel):
        self.base_model = base_model
        
    def run_scenario(self, X: pd.DataFrame, scenario_changes: Dict[str, float],
                    group_col: str = "sub_industry", n_samples: int = 1000) -> Dict[str, Any]:
        """Run scenario analysis with Bayesian uncertainty."""
        if not self.base_model.is_fitted:
            raise ValueError("Base model must be fitted before running scenarios")
        
        # Get baseline predictions
        baseline_pred = self.base_model.predict_with_intervals(X, group_col)
        
        # Apply scenario changes
        X_scenario = X.copy()
        for feature, change in scenario_changes.items():
            if feature in X_scenario.columns:
                X_scenario[feature] = X_scenario[feature] + change
        
        # Get scenario predictions
        scenario_pred = self.base_model.predict_with_intervals(X_scenario, group_col)
        
        # Calculate differences
        mean_diff = scenario_pred["mean"] - baseline_pred["mean"]
        std_diff = np.sqrt(scenario_pred["std"]**2 + baseline_pred["std"]**2)
        
        return {
            "baseline": baseline_pred,
            "scenario": scenario_pred,
            "difference": {
                "mean": mean_diff,
                "std": std_diff,
                "lower_bound": scenario_pred["lower_bound"] - baseline_pred["lower_bound"],
                "upper_bound": scenario_pred["upper_bound"] - baseline_pred["upper_bound"]
            },
            "scenario_changes": scenario_changes,
            "probability_of_improvement": self._calculate_improvement_probability(mean_diff, std_diff)
        }
    
    def _calculate_improvement_probability(self, mean_diff: np.ndarray, std_diff: np.ndarray) -> np.ndarray:
        """Calculate probability that scenario improves FVI score."""
        from scipy.stats import norm
        
        # Probability that difference > 0
        prob_improvement = 1 - norm.cdf(0, mean_diff, std_diff)
        
        return prob_improvement
    
    def monte_carlo_scenario(self, X: pd.DataFrame, scenario_distributions: Dict[str, Dict[str, float]],
                           group_col: str = "sub_industry", n_monte_carlo: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo scenario analysis."""
        logger.info(f"Running Monte Carlo scenario analysis with {n_monte_carlo} samples")
        
        baseline_pred = self.base_model.predict_with_intervals(X, group_col)
        
        # Generate scenario samples
        scenario_results = []
        
        for _ in range(n_monte_carlo):
            # Sample scenario changes
            scenario_changes = {}
            for feature, dist_params in scenario_distributions.items():
                if dist_params["type"] == "normal":
                    change = np.random.normal(dist_params["mean"], dist_params["std"])
                elif dist_params["type"] == "uniform":
                    change = np.random.uniform(dist_params["low"], dist_params["high"])
                else:
                    change = dist_params.get("value", 0)
                
                scenario_changes[feature] = change
            
            # Apply changes and predict
            X_scenario = X.copy()
            for feature, change in scenario_changes.items():
                if feature in X_scenario.columns:
                    X_scenario[feature] = X_scenario[feature] + change
            
            scenario_pred = self.base_model.predict(X_scenario, group_col)[0]  # Get mean only
            scenario_results.append({
                "prediction": scenario_pred,
                "changes": scenario_changes
            })
        
        # Analyze results
        predictions = np.array([r["prediction"] for r in scenario_results])
        
        return {
            "baseline_mean": baseline_pred["mean"],
            "scenario_mean": np.mean(predictions, axis=0),
            "scenario_std": np.std(predictions, axis=0),
            "scenario_percentiles": {
                "5th": np.percentile(predictions, 5, axis=0),
                "25th": np.percentile(predictions, 25, axis=0),
                "75th": np.percentile(predictions, 75, axis=0),
                "95th": np.percentile(predictions, 95, axis=0)
            },
            "n_scenarios": n_monte_carlo,
            "scenario_distributions": scenario_distributions
        }


# Export classes
__all__ = ["HierarchicalBayesianModel", "BayesianScenarioAnalysis"]
