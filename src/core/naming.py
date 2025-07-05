"""
FVI Naming Conventions and Constants
===================================

Strict naming patterns for all FVI artifacts to ensure consistency
across the entire system.
"""

import re
from typing import Dict, Any, Optional, Union
from enum import Enum


class FVIHorizon(Enum):
    """FVI prediction horizons."""
    H5 = "H5"   # 5 years
    H10 = "H10" # 10 years
    H20 = "H20" # 20 years


class FVINamingConventions:
    """Centralized naming conventions for FVI system."""
    
    # Patterns
    METRIC_KEY_PATTERN = r"^(\d{2})_([a-z_]+)$"
    DATASET_TABLE_PATTERN = r"^raw_([a-z_]+)_([a-z_]+)$"
    FEATURE_COLUMN_PATTERN = r"^f_(\d{2})_([a-z_]+)$"
    WEIGHT_COLUMN_PATTERN = r"^w_(\d{2})_([a-z_]+)_(H5|H10|H20)$"
    COMPOSITE_SCORE_PATTERN = r"^s_([a-z_]+)_(H5|H10|H20)$"
    
    # Sheet number mappings
    SHEET_MAPPINGS = {
        "1 Necessity Score (Core)": "01",
        "2 Resource Extraction & Scarcity Score": "02",
        "3 Artificial Support Score": "03",
        "4 Emissions Score": "04",
        "5 Ecological Score": "05",
        "8 Workforce Transition Score": "08",
        "9 Infrastructure Repurposing Score": "09",
        "11 Monopoly & Corporate Control Score": "11",
        "20 Economic Score": "20",
        "24. Technological Disruption Score": "24"
    }
    
    @staticmethod
    def generate_metric_key(sheet_name: str, slug: str) -> str:
        """Generate metric key: {sheet_number}_{slug}"""
        sheet_number = FVINamingConventions.SHEET_MAPPINGS.get(sheet_name, "00")
        clean_slug = re.sub(r'[^a-z_]', '', slug.lower().replace(' ', '_').replace('-', '_'))
        return f"{sheet_number}_{clean_slug}"
    
    @staticmethod
    def generate_dataset_table(source: str, topic: str) -> str:
        """Generate dataset table name: raw_{source}_{topic}"""
        clean_source = re.sub(r'[^a-z_]', '', source.lower().replace(' ', '_').replace('-', '_'))
        clean_topic = re.sub(r'[^a-z_]', '', topic.lower().replace(' ', '_').replace('-', '_'))
        return f"raw_{clean_source}_{clean_topic}"
    
    @staticmethod
    def generate_feature_column(metric_key: str) -> str:
        """Generate feature column name: f_{metric_key}"""
        return f"f_{metric_key}"
    
    @staticmethod
    def generate_weight_column(metric_key: str, horizon: Union[FVIHorizon, str]) -> str:
        """Generate weight column name: w_{metric_key}_{horizon}"""
        if isinstance(horizon, str):
            horizon_str = horizon
        else:
            horizon_str = horizon.value
        return f"w_{metric_key}_{horizon_str}"
    
    @staticmethod
    def generate_composite_score(subindustry: str, horizon: Union[FVIHorizon, str]) -> str:
        """Generate composite score name: s_{subindustry}_{horizon}"""
        clean_subindustry = re.sub(r'[^a-z_]', '', subindustry.lower().replace(' ', '_').replace('-', '_'))
        if isinstance(horizon, str):
            horizon_str = horizon
        else:
            horizon_str = horizon.value
        return f"s_{clean_subindustry}_{horizon_str}"
    
    @staticmethod
    def validate_metric_key(metric_key: str) -> bool:
        """Validate metric key format."""
        return bool(re.match(FVINamingConventions.METRIC_KEY_PATTERN, metric_key))
    
    @staticmethod
    def validate_dataset_table(table_name: str) -> bool:
        """Validate dataset table name format."""
        return bool(re.match(FVINamingConventions.DATASET_TABLE_PATTERN, table_name))
    
    @staticmethod
    def validate_feature_column(column_name: str) -> bool:
        """Validate feature column name format."""
        return bool(re.match(FVINamingConventions.FEATURE_COLUMN_PATTERN, column_name))
    
    @staticmethod
    def validate_weight_column(column_name: str) -> bool:
        """Validate weight column name format."""
        return bool(re.match(FVINamingConventions.WEIGHT_COLUMN_PATTERN, column_name))
    
    @staticmethod
    def validate_composite_score(score_name: str) -> bool:
        """Validate composite score name format."""
        return bool(re.match(FVINamingConventions.COMPOSITE_SCORE_PATTERN, score_name))
    
    @staticmethod
    def parse_metric_key(metric_key: str) -> Optional[Dict[str, str]]:
        """Parse metric key into components."""
        match = re.match(FVINamingConventions.METRIC_KEY_PATTERN, metric_key)
        if match:
            return {
                "sheet_number": match.group(1),
                "slug": match.group(2)
            }
        return None
    
    @staticmethod
    def parse_weight_column(column_name: str) -> Optional[Dict[str, str]]:
        """Parse weight column name into components."""
        match = re.match(FVINamingConventions.WEIGHT_COLUMN_PATTERN, column_name)
        if match:
            return {
                "sheet_number": match.group(1),
                "slug": match.group(2),
                "horizon": match.group(3)
            }
        return None
    
    @staticmethod
    def parse_composite_score(score_name: str) -> Optional[Dict[str, str]]:
        """Parse composite score name into components."""
        match = re.match(FVINamingConventions.COMPOSITE_SCORE_PATTERN, score_name)
        if match:
            return {
                "subindustry": match.group(1),
                "horizon": match.group(2)
            }
        return None


class FVIMetricFormulas:
    """Standard formulas for FVI metrics by sheet."""
    
    FORMULAS = {
        "01": {  # Necessity Score
            "energy_security_factor": "100 * (log1p(E_primary_coal_share) / log1p(100))",
            "baseload_dependency": "100 * (Coal_baseload_share / Total_baseload_capacity)",
            "economic_criticality": "100 * (Coal_GDP_contribution / Total_energy_GDP)"
        },
        "02": {  # Resource Extraction & Scarcity
            "scarcity_factor": "100 * (1 - RPR_ratio_normalised)",
            "extraction_difficulty": "100 * (1 - (Accessible_reserves / Total_reserves))",
            "grade_decline": "100 * (1 - (Current_grade / Historical_average_grade))"
        },
        "03": {  # Artificial Support
            "subsidy_factor": "100 * (Direct_Subsidies_GDPpct + Tax_Expenditure_GDPpct)",
            "policy_support": "100 * (Policy_support_score / Max_policy_score)",
            "market_distortion": "100 * (Artificial_price_support / Market_price)"
        },
        "04": {  # Emissions
            "co2_intensity": "100 * (1 - zscore(CO2_t_per_MWh))",
            "methane_factor": "100 * (1 - zscore(CH4_emissions_factor))",
            "carbon_cost_exposure": "100 * (Carbon_price_USD_per_t * Emissions_intensity)"
        },
        "05": {  # Ecological
            "land_water_impact": "100 * avg(zscore(Biodiversity_risk), zscore(Water_stress_factor))",
            "air_quality_impact": "100 * (1 - zscore(PM25_emissions_factor))",
            "ecosystem_disruption": "100 * (1 - zscore(Ecosystem_impact_score))"
        },
        "08": {  # Workforce Transition
            "reskilling_burden": "100 * sigmoid(Reskill_USD_per_worker / GDP_pc)",
            "employment_intensity": "100 * (Coal_jobs_per_GWh / Industry_avg_jobs_per_GWh)",
            "transition_cost": "100 * (Transition_cost_total / Regional_GDP)"
        },
        "09": {  # Infrastructure Repurposing
            "stranded_asset_risk": "100 * (1 - (Years_remaining / Nominal_life))",
            "repurposing_potential": "100 * (Repurposing_score / Max_repurposing_score)",
            "infrastructure_age": "100 * (Current_age / Design_life)"
        },
        "11": {  # Monopoly & Corporate Control
            "hhi_dominance": "100 * (HHI / 10000)",
            "market_concentration": "100 * (Top4_market_share / 100)",
            "pricing_power": "100 * (Price_markup / Competitive_benchmark)"
        },
        "20": {  # Economic
            "profit_resilience": "100 * (zscore(EBIT_margin) + zscore(Price_volatility)) / 2",
            "capex_intensity": "100 * (1 - zscore(CAPEX_per_MW))",
            "opex_competitiveness": "100 * (1 - zscore(OPEX_per_MWh))"
        },
        "24": {  # Technological Disruption
            "substitute_maturity": "100 * (1 - (LCOE_coal / LCOE_alternatives))",
            "innovation_rate": "100 * (1 - zscore(Patent_growth_rate))",
            "technology_gap": "100 * (1 - (Performance_gap / Historical_gap))"
        }
    }
    
    @staticmethod
    def get_formula(sheet_number: str, metric_slug: str) -> Optional[str]:
        """Get formula for a specific metric."""
        sheet_formulas = FVIMetricFormulas.FORMULAS.get(sheet_number, {})
        return sheet_formulas.get(metric_slug)
    
    @staticmethod
    def get_sheet_formulas(sheet_number: str) -> Dict[str, str]:
        """Get all formulas for a sheet."""
        return FVIMetricFormulas.FORMULAS.get(sheet_number, {})
    
    @staticmethod
    def list_all_formulas() -> Dict[str, Dict[str, str]]:
        """Get all formulas."""
        return FVIMetricFormulas.FORMULAS


class FVIDataQualityWeights:
    """Data quality components for dynamic weighting."""
    
    QUALITY_COMPONENTS = {
        "freshness": 0.4,      # How recent is the data
        "coverage": 0.3,       # Geographic/temporal coverage
        "confidence": 0.3      # Data reliability/accuracy
    }
    
    @staticmethod
    def calculate_dq_score(freshness: float, coverage: float, confidence: float) -> float:
        """Calculate data quality score (0-1)."""
        weights = FVIDataQualityWeights.QUALITY_COMPONENTS
        return (
            weights["freshness"] * freshness +
            weights["coverage"] * coverage +
            weights["confidence"] * confidence
        )
    
    @staticmethod
    def adjust_weight(prior_weight: float, dq_score: float) -> float:
        """Adjust weight based on data quality."""
        return prior_weight * dq_score


# Export main classes
__all__ = [
    "FVINamingConventions",
    "FVIHorizon", 
    "FVIMetricFormulas",
    "FVIDataQualityWeights"
]
