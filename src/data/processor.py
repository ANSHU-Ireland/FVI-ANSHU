import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import re
from slugify import slugify

from ..models.schemas import DataSourceCreate, MetricDefinitionCreate
from ..config import settings

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Process the FVI Excel workbook and extract metric definitions."""
    
    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.workbook = None
        self.sheet_names = []
        
    def load_workbook(self) -> None:
        """Load the Excel workbook."""
        try:
            self.workbook = pd.ExcelFile(self.excel_path)
            self.sheet_names = self.workbook.sheet_names
            logger.info(f"Loaded workbook with {len(self.sheet_names)} sheets")
        except Exception as e:
            logger.error(f"Error loading workbook: {e}")
            raise
    
    def get_sheet_mapping(self) -> Dict[str, Dict[str, str]]:
        """Get the mapping of sheet names to their thematic focus."""
        sheet_mapping = {
            "1 Necessity Score (Core)": {
                "theme": "Social & economic indispensability of coal today",
                "type": "metric_definition"
            },
            "2 Resource Extraction & Scarcity Score": {
                "theme": "How hard & costly it is to obtain coal going forward",
                "type": "metric_definition"
            },
            "3 Artificial Support Score": {
                "theme": "Degree of subsidies, bail-outs, mandates keeping coal alive",
                "type": "metric_definition"
            },
            "4 Emissions Score": {
                "theme": "GHG impact & regulatory exposure",
                "type": "metric_definition"
            },
            "5 Ecological Score": {
                "theme": "Wider environmental externalities (land, water, biodiversity)",
                "type": "metric_definition"
            },
            "8 Workforce Transition Score": {
                "theme": "Reskilling cost & labour-market rigidity",
                "type": "metric_definition"
            },
            "9 Infrastructure Repurposing Score": {
                "theme": "How easily coal assets can be repurposed",
                "type": "metric_definition"
            },
            "11 Monopoly & Corporate Control Score": {
                "theme": "Market structure & pricing power",
                "type": "metric_definition"
            },
            "20 Economic Score": {
                "theme": "Profitability & macro-exposure (prices, GDP elasticity)",
                "type": "metric_definition"
            },
            "24. Technological Disruption Score": {
                "theme": "Threat of substitutes (e.g. renewables, CCS)",
                "type": "metric_definition"
            },
            "Data Sources": {
                "theme": "Catalogue of every data asset referenced above",
                "type": "reference_table"
            }
        }
        return sheet_mapping
    
    def process_data_sources_sheet(self) -> List[DataSourceCreate]:
        """Process the Data Sources sheet."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name="Data Sources")
            data_sources = []
            
            # Expected columns in Data Sources sheet
            column_mapping = {
                "Name": "name",
                "Source Link": "source_link",
                "Description": "description",
                "LV 3 GICS Schema": "gics_schema",
                "Frequency": "frequency",
                "Currency": "currency",
                "Update Lag_days": "update_lag_days",
                "Type": "data_type",
                "Data Format": "data_format",
                "Region": "region",
                "Country": "country",
                "Category": "category",
                "Sub-Category": "sub_category",
                "Lifespan": "lifespan",
                "API": "api_available",
                "Github": "github_repo",
                "File Format": "file_format",
                "File": "file_path",
                "Source": "source_author",
                "License": "license_type",
                "NON-GICS Sector": "non_gics_sector"
            }
            
            for _, row in df.iterrows():
                data_source_dict = {}
                for excel_col, pydantic_field in column_mapping.items():
                    if excel_col in df.columns:
                        value = row[excel_col]
                        if pd.notna(value):
                            if pydantic_field == "api_available":
                                data_source_dict[pydantic_field] = str(value).lower() in ['true', 'yes', '1', 'y']
                            elif pydantic_field == "update_lag_days":
                                try:
                                    data_source_dict[pydantic_field] = int(value)
                                except (ValueError, TypeError):
                                    data_source_dict[pydantic_field] = None
                            else:
                                data_source_dict[pydantic_field] = str(value)
                
                if data_source_dict.get("name"):
                    data_sources.append(DataSourceCreate(**data_source_dict))
            
            logger.info(f"Processed {len(data_sources)} data sources")
            return data_sources
            
        except Exception as e:
            logger.error(f"Error processing Data Sources sheet: {e}")
            return []
    
    def process_metric_definition_sheet(self, sheet_name: str) -> List[MetricDefinitionCreate]:
        """Process a metric definition sheet."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            metrics = []
            
            # Expected columns in metric definition sheets
            column_mapping = {
                "ID Number": "id_number",
                "Title": "title",
                "METRIC": "metric_slug",
                "Details": "details",
                "Data Fields required": "data_fields_required",
                "Formula": "formula",
                "Weighting In Metric": "weighting_in_metric",
                "Data Source": "data_source_name",
                "Structured": "structured_availability",
                "Unstructured": "unstructured_availability",
                "Country-Level": "country_level_availability",
                "Sub-Industry Availability": "sub_industry_availability",
                "Volume of Data": "volume_of_data",
                "Alternative Proxy Feasibility": "alternative_proxy_feasibility",
                "GenAI / ML Fillability": "genai_ml_fillability",
                "Industry-Level Variability": "industry_level_variability",
                "Longitudinal Availability": "longitudinal_availability",
                "Data Verification & Bias Risk": "data_verification_bias_risk",
                "Interdependence with Other Metrics": "interdependence_with_other_metrics",
                "Overlap with Other Metrics": "overlap_with_other_metrics",
                "Scoring Methodology Notes": "scoring_methodology_notes",
                "Readiness Status": "readiness_status",
                "Linked Sheet or Tab": "linked_sheet_or_tab"
            }
            
            for _, row in df.iterrows():
                metric_dict = {"sheet_name": sheet_name}
                
                for excel_col, pydantic_field in column_mapping.items():
                    if excel_col in df.columns:
                        value = row[excel_col]
                        if pd.notna(value):
                            if pydantic_field in ["weighting_in_metric"]:
                                try:
                                    metric_dict[pydantic_field] = float(value)
                                except (ValueError, TypeError):
                                    metric_dict[pydantic_field] = None
                            elif pydantic_field in [
                                "structured_availability", "unstructured_availability",
                                "country_level_availability", "sub_industry_availability",
                                "volume_of_data", "alternative_proxy_feasibility",
                                "genai_ml_fillability", "industry_level_variability",
                                "longitudinal_availability", "data_verification_bias_risk",
                                "interdependence_with_other_metrics"
                            ]:
                                try:
                                    metric_dict[pydantic_field] = int(value)
                                except (ValueError, TypeError):
                                    metric_dict[pydantic_field] = None
                            else:
                                metric_dict[pydantic_field] = str(value)
                
                # Generate metric slug if not provided
                if not metric_dict.get("metric_slug") and metric_dict.get("title"):
                    metric_dict["metric_slug"] = slugify(metric_dict["title"], separator="_")
                
                # Skip rows without title
                if metric_dict.get("title"):
                    metrics.append(MetricDefinitionCreate(**metric_dict))
            
            logger.info(f"Processed {len(metrics)} metrics from sheet '{sheet_name}'")
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing metric sheet '{sheet_name}': {e}")
            return []
    
    def process_all_sheets(self) -> Dict[str, Any]:
        """Process all sheets in the workbook."""
        if not self.workbook:
            self.load_workbook()
        
        results = {
            "data_sources": [],
            "metrics": [],
            "sheet_mapping": self.get_sheet_mapping()
        }
        
        # Process Data Sources sheet
        if "Data Sources" in self.sheet_names:
            results["data_sources"] = self.process_data_sources_sheet()
        
        # Process metric definition sheets
        metric_sheets = [
            name for name in self.sheet_names 
            if name in results["sheet_mapping"] and 
            results["sheet_mapping"][name]["type"] == "metric_definition"
        ]
        
        for sheet_name in metric_sheets:
            metrics = self.process_metric_definition_sheet(sheet_name)
            results["metrics"].extend(metrics)
        
        logger.info(f"Processed {len(results['data_sources'])} data sources and {len(results['metrics'])} metrics")
        return results
    
    def extract_special_columns(self, sheet_name: str) -> Dict[str, Any]:
        """Extract special columns from specific sheets."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            special_data = {}
            
            if sheet_name == "4 Emissions Score":
                # Extract emissions-specific columns
                emissions_columns = ["CO₂e_tonnes", "Output_or_GlobalTotal_or_Coverage_pct", "Employees", "Value_Added_USD"]
                for col in emissions_columns:
                    if col in df.columns:
                        special_data[col] = df[col].tolist()
            
            elif sheet_name == "20 Economic Score":
                # Extract economic-specific columns
                if "Variable Data Fields" in df.columns:
                    special_data["variable_data_fields"] = df["Variable Data Fields"].tolist()
            
            # Extract embedded data from Details column using regex
            if "Details" in df.columns:
                special_data["extracted_references"] = self._extract_references_from_details(df["Details"])
            
            return special_data
            
        except Exception as e:
            logger.error(f"Error extracting special columns from '{sheet_name}': {e}")
            return {}
    
    def _extract_references_from_details(self, details_series: pd.Series) -> List[Dict[str, str]]:
        """Extract references from the Details column using regex."""
        references = []
        
        # Common patterns to extract
        patterns = {
            "usgs_data": r"USGS.*?R/P Ratio",
            "oecd_subsidy": r"OECD.*?ETP.*?Subsidy",
            "imf_tax": r"IMF.*?Tax Expenditure",
            "reskilling_cost": r"Reskilling Cost.*?(\d+).*?USD",
            "unionisation_rate": r"Unionisation.*?Rate.*?(\d+\.?\d*)%",
            "asset_life": r"Asset Life.*?Remaining.*?(\d+).*?years",
            "hhi_current": r"HHI.*?current.*?(\d+\.?\d*)",
            "learning_rate": r"Learning Rate.*?(\d+\.?\d*)%",
            "patents_cagr": r"Patents.*?FiveYearCAGR.*?(\d+\.?\d*)%"
        }
        
        for detail in details_series:
            if pd.notna(detail):
                detail_str = str(detail)
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, detail_str, re.IGNORECASE)
                    if matches:
                        references.append({
                            "type": pattern_name,
                            "source_text": detail_str,
                            "extracted_values": matches
                        })
        
        return references


class DataNormalizer:
    """Normalize and clean data for FVI calculations."""
    
    @staticmethod
    def normalize_to_0_100(values: List[float], method: str = "min_max") -> List[float]:
        """Normalize values to 0-100 scale."""
        if not values:
            return []
        
        values = np.array(values)
        
        if method == "min_max":
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val == min_val:
                return [50.0] * len(values)  # Return middle value if all same
            normalized = (values - min_val) / (max_val - min_val) * 100
        
        elif method == "z_score":
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val == 0:
                return [50.0] * len(values)
            z_scores = (values - mean_val) / std_val
            # Convert z-scores to 0-100 scale (assuming ~99.7% within 3 std devs)
            normalized = np.clip((z_scores + 3) / 6 * 100, 0, 100)
        
        elif method == "percentile":
            percentiles = [np.percentile(values, p) for p in np.linspace(0, 100, len(values))]
            normalized = np.array(percentiles)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized.tolist()
    
    @staticmethod
    def handle_missing_values(values: List[Optional[float]], method: str = "mean") -> List[float]:
        """Handle missing values in data."""
        if not values:
            return []
        
        # Convert to numpy array and handle None values
        arr = np.array([v if v is not None else np.nan for v in values])
        
        if method == "mean":
            fill_value = np.nanmean(arr)
        elif method == "median":
            fill_value = np.nanmedian(arr)
        elif method == "forward_fill":
            fill_value = None  # Special case
        elif method == "zero":
            fill_value = 0.0
        else:
            raise ValueError(f"Unknown missing value method: {method}")
        
        if method == "forward_fill":
            # Forward fill missing values
            last_valid = None
            for i in range(len(arr)):
                if not np.isnan(arr[i]):
                    last_valid = arr[i]
                elif last_valid is not None:
                    arr[i] = last_valid
                else:
                    arr[i] = 0.0  # Fill with zero if no previous value
        else:
            # Fill with computed value
            if not np.isnan(fill_value):
                arr = np.where(np.isnan(arr), fill_value, arr)
            else:
                arr = np.where(np.isnan(arr), 0.0, arr)
        
        return arr.tolist()
    
    @staticmethod
    def detect_outliers(values: List[float], method: str = "iqr") -> List[int]:
        """Detect outliers in data and return their indices."""
        if not values:
            return []
        
        values = np.array(values)
        outlier_indices = []
        
        if method == "iqr":
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = np.where(
                (values < lower_bound) | (values > upper_bound)
            )[0].tolist()
        
        elif method == "z_score":
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outlier_indices = np.where(z_scores > 3)[0].tolist()
        
        elif method == "modified_z_score":
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > 3.5)[0].tolist()
        
        return outlier_indices
    
    @staticmethod
    def apply_formula(data: Dict[str, List[float]], formula: str) -> List[float]:
        """Apply a formula to data fields."""
        try:
            # Simple formula parser for basic mathematical operations
            # This is a simplified version - in production, use a proper expression parser
            
            # Replace field names with actual data
            formula_code = formula
            for field_name, field_data in data.items():
                formula_code = formula_code.replace(field_name, f"data['{field_name}']")
            
            # Evaluate the formula
            result = eval(formula_code)
            
            if isinstance(result, (int, float)):
                return [result] * len(next(iter(data.values())))
            elif isinstance(result, list):
                return result
            else:
                return [float(result)] * len(next(iter(data.values())))
            
        except Exception as e:
            logger.error(f"Error applying formula '{formula}': {e}")
            return [0.0] * len(next(iter(data.values())))


# Export classes
__all__ = ["ExcelProcessor", "DataNormalizer"]
