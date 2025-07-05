#!/usr/bin/env python3
"""
Excel to YAML Converter for FVI Metrics
=======================================

Converts the FVI Excel workbook to a structured YAML catalog
following the naming conventions and data structure requirements.

Usage:
    python scripts/excel2yaml.py "data/FVI Scoring Metrics_Coal.xlsx" --out meta/metric_catalogue.yaml
"""

import argparse
import yaml
import pandas as pd
import re
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.naming import FVINamingConventions, FVIMetricFormulas

logger = logging.getLogger(__name__)


class ExcelToYAMLConverter:
    """Convert Excel workbook to YAML metric catalog."""
    
    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.workbook = None
        self.catalog = {
            "metadata": {
                "name": "FVI Coal Metrics Catalog",
                "version": "1.0.0",
                "description": "Future Viability Index metrics for Coal industry",
                "created_from": str(self.excel_path),
                "naming_convention": "FVI Standard v1.0"
            },
            "data_sources": [],
            "metric_definitions": [],
            "sheet_mappings": FVINamingConventions.SHEET_MAPPINGS
        }
    
    def load_workbook(self) -> None:
        """Load the Excel workbook."""
        try:
            self.workbook = pd.ExcelFile(self.excel_path)
            logger.info(f"Loaded workbook: {self.excel_path}")
        except Exception as e:
            logger.error(f"Error loading workbook: {e}")
            raise
    
    def convert_data_sources(self) -> List[Dict[str, Any]]:
        """Convert Data Sources sheet to YAML format."""
        try:
            if "Data Sources" not in self.workbook.sheet_names:
                logger.warning("No 'Data Sources' sheet found")
                return []
            
            df = pd.read_excel(self.excel_path, sheet_name="Data Sources")
            data_sources = []
            
            for _, row in df.iterrows():
                if pd.notna(row.get("Name")):
                    data_source = {
                        "name": str(row["Name"]),
                        "source_link": str(row.get("Source Link", "")) if pd.notna(row.get("Source Link")) else None,
                        "description": str(row.get("Description", "")) if pd.notna(row.get("Description")) else None,
                        "metadata": {
                            "gics_schema": str(row.get("LV 3 GICS Schema", "")) if pd.notna(row.get("LV 3 GICS Schema")) else None,
                            "frequency": str(row.get("Frequency", "")) if pd.notna(row.get("Frequency")) else None,
                            "currency": str(row.get("Currency", "")) if pd.notna(row.get("Currency")) else None,
                            "update_lag_days": int(row.get("Update Lag_days", 0)) if pd.notna(row.get("Update Lag_days")) else None,
                            "data_type": str(row.get("Type", "")) if pd.notna(row.get("Type")) else None,
                            "data_format": str(row.get("Data Format", "")) if pd.notna(row.get("Data Format")) else None,
                            "region": str(row.get("Region", "")) if pd.notna(row.get("Region")) else None,
                            "country": str(row.get("Country", "")) if pd.notna(row.get("Country")) else None,
                            "category": str(row.get("Category", "")) if pd.notna(row.get("Category")) else None,
                            "sub_category": str(row.get("Sub-Category", "")) if pd.notna(row.get("Sub-Category")) else None,
                            "lifespan": str(row.get("Lifespan", "")) if pd.notna(row.get("Lifespan")) else None,
                            "api_available": bool(str(row.get("API", "")).lower() in ["true", "yes", "1"]) if pd.notna(row.get("API")) else False,
                            "github_repo": str(row.get("Github", "")) if pd.notna(row.get("Github")) else None,
                            "file_format": str(row.get("File Format", "")) if pd.notna(row.get("File Format")) else None,
                            "file_path": str(row.get("File", "")) if pd.notna(row.get("File")) else None,
                            "source_author": str(row.get("Source", "")) if pd.notna(row.get("Source")) else None,
                            "license_type": str(row.get("License", "")) if pd.notna(row.get("License")) else None,
                            "non_gics_sector": str(row.get("NON-GICS Sector", "")) if pd.notna(row.get("NON-GICS Sector")) else None
                        }
                    }
                    
                    # Clean up None values
                    data_source["metadata"] = {k: v for k, v in data_source["metadata"].items() if v is not None}
                    
                    data_sources.append(data_source)
            
            logger.info(f"Converted {len(data_sources)} data sources")
            return data_sources
            
        except Exception as e:
            logger.error(f"Error converting data sources: {e}")
            return []
    
    def convert_metric_definitions(self) -> List[Dict[str, Any]]:
        """Convert metric definition sheets to YAML format."""
        metric_definitions = []
        
        for sheet_name in self.workbook.sheet_names:
            if sheet_name in FVINamingConventions.SHEET_MAPPINGS:
                try:
                    df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                    sheet_number = FVINamingConventions.SHEET_MAPPINGS[sheet_name]
                    
                    for _, row in df.iterrows():
                        # Convert row values to handle different types
                        row_dict = {}
                        for key, value in row.items():
                            if pd.notna(value):
                                # Handle different types of values
                                if hasattr(value, 'value'):
                                    row_dict[key] = value.value
                                else:
                                    row_dict[key] = value
                            else:
                                row_dict[key] = None
                        
                        if row_dict.get("Title"):
                            # Generate standardized names
                            title = str(row_dict["Title"])
                            slug = title.lower().replace(" ", "_").replace("-", "_")
                            metric_key = FVINamingConventions.generate_metric_key(sheet_name, slug)
                            feature_column = FVINamingConventions.generate_feature_column(metric_key)
                            
                            # Get formula from formula catalog
                            formula = FVIMetricFormulas.get_formula(sheet_number, slug)
                            
                            metric_def = {
                                "metric_key": metric_key,
                                "feature_column": feature_column,
                                "sheet_info": {
                                    "sheet_name": sheet_name,
                                    "sheet_number": sheet_number,
                                    "thematic_focus": self._get_sheet_theme(sheet_name)
                                },
                                "basic_info": {
                                    "title": title,
                                    "slug": slug,
                                    "details": str(row_dict.get("Details", "")) if row_dict.get("Details") else None,
                                    "data_fields_required": str(row_dict.get("Data Fields required", "")) if row_dict.get("Data Fields required") else None,
                                    "formula": formula or str(row_dict.get("Formula", "")) if row_dict.get("Formula") else None,
                                    "weighting_in_metric": self._safe_numeric_conversion(row_dict.get("Weighting In Metric"), float, 0.0),
                                    "data_source": str(row_dict.get("Data Source", "")) if row_dict.get("Data Source") else None
                                },
                                "data_quality": {
                                    "structured_availability": self._safe_numeric_conversion(row_dict.get("Structured"), int, 0),
                                    "unstructured_availability": self._safe_numeric_conversion(row_dict.get("Unstructured"), int, 0),
                                    "country_level_availability": self._safe_numeric_conversion(row_dict.get("Country-Level"), int, 0),
                                    "sub_industry_availability": self._safe_numeric_conversion(row_dict.get("Sub-Industry Availability"), int, 0),
                                    "volume_of_data": self._safe_numeric_conversion(row_dict.get("Volume of Data"), int, 0),
                                    "alternative_proxy_feasibility": self._safe_numeric_conversion(row_dict.get("Alternative Proxy Feasibility"), int, 0),
                                    "genai_ml_fillability": self._safe_numeric_conversion(row_dict.get("GenAI / ML Fillability"), int, 0),
                                    "industry_level_variability": self._safe_numeric_conversion(row_dict.get("Industry-Level Variability"), int, 0),
                                    "longitudinal_availability": self._safe_numeric_conversion(row_dict.get("Longitudinal Availability"), int, 0),
                                    "data_verification_bias_risk": self._safe_numeric_conversion(row_dict.get("Data Verification & Bias Risk"), int, 0),
                                    "interdependence_with_other_metrics": self._safe_numeric_conversion(row_dict.get("Interdependence with Other Metrics"), int, 0)
                                },
                                "assessment": {
                                    "overlap_with_other_metrics": str(row_dict.get("Overlap with Other Metrics", "")) if row_dict.get("Overlap with Other Metrics") else None,
                                    "scoring_methodology_notes": str(row_dict.get("Scoring Methodology Notes", "")) if row_dict.get("Scoring Methodology Notes") else None,
                                    "readiness_status": str(row_dict.get("Readiness Status", "Draft")) if row_dict.get("Readiness Status") else "Draft",
                                    "linked_sheet_or_tab": str(row_dict.get("Linked Sheet or Tab", "")) if row_dict.get("Linked Sheet or Tab") else None
                                },
                                "weight_columns": {
                                    "H5": FVINamingConventions.generate_weight_column(metric_key, "H5"),
                                    "H10": FVINamingConventions.generate_weight_column(metric_key, "H10"),
                                    "H20": FVINamingConventions.generate_weight_column(metric_key, "H20")
                                }
                            }
                            
                            # Clean up None values
                            metric_def = self._clean_dict(metric_def)
                            metric_definitions.append(metric_def)
                    
                    logger.info(f"Converted {len([m for m in metric_definitions if m['sheet_info']['sheet_name'] == sheet_name])} metrics from {sheet_name}")
                    
                except Exception as e:
                    logger.error(f"Error converting sheet {sheet_name}: {e}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        logger.info(f"Converted {len(metric_definitions)} total metric definitions")
        return metric_definitions
    
    def _get_sheet_theme(self, sheet_name: str) -> str:
        """Get thematic focus for a sheet."""
        themes = {
            "1 Necessity Score (Core)": "Social & economic indispensability of coal today",
            "2 Resource Extraction & Scarcity Score": "How hard & costly it is to obtain coal going forward",
            "3 Artificial Support Score": "Degree of subsidies, bail-outs, mandates keeping coal alive",
            "4 Emissions Score": "GHG impact & regulatory exposure",
            "5 Ecological Score": "Wider environmental externalities (land, water, biodiversity)",
            "8 Workforce Transition Score": "Reskilling cost & labour-market rigidity",
            "9 Infrastructure Repurposing Score": "How easily coal assets can be repurposed",
            "11 Monopoly & Corporate Control Score": "Market structure & pricing power",
            "20 Economic Score": "Profitability & macro-exposure (prices, GDP elasticity)",
            "24. Technological Disruption Score": "Threat of substitutes (e.g. renewables, CCS)"
        }
        return themes.get(sheet_name, "Unknown")

    def _safe_numeric_conversion(self, value: Any, conversion_type: type, default: Any = None) -> Any:
        """Safely convert values to numeric types."""
        if value is None:
            return default
        
        try:
            # Clean up string values
            if isinstance(value, str):
                # Remove common text patterns
                value = re.sub(r'\s*\([^)]+\)\s*', '', value)  # Remove parenthetical notes
                value = value.strip()
                
                # If it's empty after cleaning, return default
                if not value:
                    return default
            
            return conversion_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to {conversion_type.__name__}, using default {default}")
            return default
    
    def _safe_numeric_conversion(self, value: Any, conversion_type: type, default: Any = None) -> Any:
        """Safely convert values to numeric types."""
        if value is None:
            return default
        
        try:
            # Clean up string values
            if isinstance(value, str):
                # Remove common text patterns
                value = re.sub(r'\s*\([^)]+\)\s*', '', value)  # Remove parenthetical notes
                value = value.strip()
                
                # If it's empty after cleaning, return default
                if not value:
                    return default
            
            return conversion_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to {conversion_type.__name__}, using default {default}")
            return default

    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively clean dictionary of None values."""
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [self._clean_dict(item) for item in d if item is not None]
        else:
            return d
    
    def convert_to_yaml(self) -> Dict[str, Any]:
        """Convert entire workbook to YAML catalog."""
        if not self.workbook:
            self.load_workbook()
        
        # Convert data sources
        self.catalog["data_sources"] = self.convert_data_sources()
        
        # Convert metric definitions
        self.catalog["metric_definitions"] = self.convert_metric_definitions()
        
        # Add summary statistics
        self.catalog["summary"] = {
            "total_data_sources": len(self.catalog["data_sources"]),
            "total_metrics": len(self.catalog["metric_definitions"]),
            "sheets_processed": len(self.catalog["sheet_mappings"]),
            "metrics_by_sheet": {}
        }
        
        # Count metrics by sheet
        for metric in self.catalog["metric_definitions"]:
            sheet_name = metric["sheet_info"]["sheet_name"]
            if sheet_name not in self.catalog["summary"]["metrics_by_sheet"]:
                self.catalog["summary"]["metrics_by_sheet"][sheet_name] = 0
            self.catalog["summary"]["metrics_by_sheet"][sheet_name] += 1
        
        return self.catalog
    
    def save_yaml(self, output_path: str) -> None:
        """Save catalog to YAML file."""
        catalog = self.convert_to_yaml()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(catalog, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Saved YAML catalog to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert FVI Excel workbook to YAML catalog")
    parser.add_argument("excel_file", help="Path to Excel file")
    parser.add_argument("--out", default="meta/metric_catalogue.yaml", help="Output YAML file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Check if input file exists
        if not Path(args.excel_file).exists():
            logger.error(f"Excel file not found: {args.excel_file}")
            return 1
        
        # Convert Excel to YAML
        converter = ExcelToYAMLConverter(args.excel_file)
        converter.save_yaml(args.out)
        
        logger.info("Conversion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
