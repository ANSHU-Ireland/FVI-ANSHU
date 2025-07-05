#!/usr/bin/env python3
"""
Script to process FVI Excel data and populate the database.
"""
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the src directory to path
sys.path.append('/workspaces/FVI-ANSHU/src')

from database.connection import DatabaseConnection
from database.crud import DatabaseCRUD
from models.schemas import DataSourceCreate, MetricDefinitionCreate

class FVIDataProcessor:
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.db_conn = DatabaseConnection()
        self.crud = DatabaseCRUD(self.db_conn)
        
    def process_data_sources_sheet(self) -> List[Dict[str, Any]]:
        """Process the Data Sources sheet."""
        print("Processing Data Sources sheet...")
        
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name='Data Sources')
            
            # Clean and process data sources
            data_sources = []
            for idx, row in df.iterrows():
                if pd.notna(row.get('Name', '')):
                    source_data = {
                        'name': str(row.get('Name', '')).strip(),
                        'url': str(row.get('Source Link', '')) if pd.notna(row.get('Source Link', '')) else None,
                        'description': str(row.get('Description', '')) if pd.notna(row.get('Description', '')) else None,
                        'source_type': str(row.get('Type', 'External')) if pd.notna(row.get('Type', '')) else 'External',
                        'api_available': bool(row.get('API', False)) if pd.notna(row.get('API', False)) else False,
                        'data_format': str(row.get('Data Format', '')) if pd.notna(row.get('Data Format', '')) else None,
                        'region': str(row.get('Region', '')) if pd.notna(row.get('Region', '')) else None,
                        'category': str(row.get('Category', '')) if pd.notna(row.get('Category', '')) else None,
                        'sub_category': str(row.get('Sub-Category', '')) if pd.notna(row.get('Sub-Category', '')) else None,
                        'author': str(row.get('Author', '')) if pd.notna(row.get('Author', '')) else None,
                        'license': str(row.get('License', '')) if pd.notna(row.get('License', '')) else None,
                        'is_active': True,
                        'metadata': {
                            'gics_sector': str(row.get('Sector (from LV 3 GICS Schema Mar2023)', '')) if pd.notna(row.get('Sector (from LV 3 GICS Schema Mar2023)', '')) else None,
                            'industry_group': str(row.get('IndustryGroup (from LV 3 GICS Schema Mar2023)', '')) if pd.notna(row.get('IndustryGroup (from LV 3 GICS Schema Mar2023)', '')) else None,
                            'industry': str(row.get('Industry (from LV 3 GICS Schema Mar2023)', '')) if pd.notna(row.get('Industry (from LV 3 GICS Schema Mar2023)', '')) else None,
                            'sub_industry': str(row.get('SubIndustry (from LV 3 GICS Schema Mar2023)', '')) if pd.notna(row.get('SubIndustry (from LV 3 GICS Schema Mar2023)', '')) else None,
                            'lifespan': str(row.get('Lifespan', '')) if pd.notna(row.get('Lifespan', '')) else None,
                            'file_format': str(row.get('File Format', '')) if pd.notna(row.get('File Format', '')) else None,
                            'source': str(row.get('Source', '')) if pd.notna(row.get('Source', '')) else None,
                            'dg_internal_note': str(row.get('D&G Internal Note', '')) if pd.notna(row.get('D&G Internal Note', '')) else None,
                            'dg_product': str(row.get('D&G Product', '')) if pd.notna(row.get('D&G Product', '')) else None,
                            'non_gics_sector': str(row.get('NON-GICS Sector', '')) if pd.notna(row.get('NON-GICS Sector', '')) else None
                        }
                    }
                    data_sources.append(source_data)
            
            print(f"Processed {len(data_sources)} data sources")
            return data_sources
            
        except Exception as e:
            print(f"Error processing data sources: {e}")
            return []
    
    def process_metric_sheet(self, sheet_name: str) -> List[Dict[str, Any]]:
        """Process a metric definition sheet."""
        print(f"Processing sheet: {sheet_name}")
        
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name)
            
            # Extract composite score name from sheet name
            if sheet_name.startswith('1 '):
                composite_score = 'Necessity Score'
            elif sheet_name.startswith('2 '):
                composite_score = 'Resource Extraction & Scarcity Score'
            elif sheet_name.startswith('3 '):
                composite_score = 'Artificial Support Score'
            elif sheet_name.startswith('4 '):
                composite_score = 'Emissions Score'
            elif sheet_name.startswith('5 '):
                composite_score = 'Ecological Score'
            elif sheet_name.startswith('8 '):
                composite_score = 'Workforce Transition Score'
            elif sheet_name.startswith('9 '):
                composite_score = 'Infrastructure Repurposing Score'
            elif sheet_name.startswith('11 '):
                composite_score = 'Monopoly & Corporate Control Score'
            elif sheet_name.startswith('20 '):
                composite_score = 'Economic Score'
            elif sheet_name.startswith('24'):
                composite_score = 'Technological Disruption Score'
            else:
                composite_score = sheet_name
            
            metrics = []
            for idx, row in df.iterrows():
                # Skip empty rows
                if pd.isna(row.get('METRIC', '')) and pd.isna(row.get('Title', '')):
                    continue
                
                # Skip header rows
                if str(row.get('METRIC', '')).upper() == 'METRIC':
                    continue
                
                metric_name = str(row.get('METRIC', '')) if pd.notna(row.get('METRIC', '')) else None
                metric_title = str(row.get('Title', '')) if pd.notna(row.get('Title', '')) else None
                
                if metric_name or metric_title:
                    metric_data = {
                        'name': metric_name or metric_title or f"Metric_{idx}",
                        'title': metric_title,
                        'description': str(row.get('Details', '')) if pd.notna(row.get('Details', '')) else None,
                        'category': composite_score,
                        'industry': 'Coal',
                        'formula': str(row.get('Formula', '')) if pd.notna(row.get('Formula', '')) else None,
                        'data_sources': [str(row.get('Data Source', ''))] if pd.notna(row.get('Data Source', '')) else [],
                        'weight': float(row.get('Weighting In Metric', 1.0)) if pd.notna(row.get('Weighting In Metric', 1.0)) else 1.0,
                        'is_active': True,
                        'metadata': {
                            'id_number': str(row.get('ID Number', '')) if pd.notna(row.get('ID Number', '')) else None,
                            'composite_score': str(row.get('COMPOSITE SCORE', '')) if pd.notna(row.get('COMPOSITE SCORE', '')) else None,
                            'data_fields_required': str(row.get('Data Fields required', '')) if pd.notna(row.get('Data Fields required', '')) else None,
                            'structured_data_availability': float(row.get('Structured Data Availability (1–5)', 0)) if pd.notna(row.get('Structured Data Availability (1–5)', 0)) else None,
                            'unstructured_data_availability': float(row.get('Unstructured Data Availability (1–5)', 0)) if pd.notna(row.get('Unstructured Data Availability (1–5)', 0)) else None,
                            'country_level_data_availability': float(row.get('Country-Level Data Availability (1–5)', 0)) if pd.notna(row.get('Country-Level Data Availability (1–5)', 0)) else None,
                            'gics_sub_industry_data_availability': float(row.get('GICS Sub-Industry Data Availability (1–5)', 0)) if pd.notna(row.get('GICS Sub-Industry Data Availability (1–5)', 0)) else None,
                            'volume_of_data': float(row.get('Volume of Data (1–5)', 0)) if pd.notna(row.get('Volume of Data (1–5)', 0)) else None,
                            'alternative_proxy_feasibility': float(row.get('Alternative Proxy Feasibility (1–5)', 0)) if pd.notna(row.get('Alternative Proxy Feasibility (1–5)', 0)) else None,
                            'genai_ml_fillability': float(row.get('GenAI / ML Fillability (1–5)', 0)) if pd.notna(row.get('GenAI / ML Fillability (1–5)', 0)) else None,
                            'industry_level_variability': float(row.get('Industry-Level Variability (1–5)', 0)) if pd.notna(row.get('Industry-Level Variability (1–5)', 0)) else None,
                            'longitudinal_data_availability': float(row.get('Longitudinal Data Availability (1–5)', 0)) if pd.notna(row.get('Longitudinal Data Availability (1–5)', 0)) else None,
                            'data_verification_bias_risk': float(row.get('Data Verification & Bias Risk (1–5)', 0)) if pd.notna(row.get('Data Verification & Bias Risk (1–5)', 0)) else None,
                            'interdependence_with_other_metrics': float(row.get('Interdependence with Other Metrics (1–5)', 0)) if pd.notna(row.get('Interdependence with Other Metrics (1–5)', 0)) else None,
                            'overlap_with_other_metrics': str(row.get('Overlap with Other Metrics (High/Medium/Low)', '')) if pd.notna(row.get('Overlap with Other Metrics (High/Medium/Low)', '')) else None,
                            'final_data_confidence_score': float(row.get('Final Data Confidence Score (Auto, 1–5)', 0)) if pd.notna(row.get('Final Data Confidence Score (Auto, 1–5)', 0)) else None,
                            'scoring_methodology_notes': str(row.get('Scoring Methodology Notes', '')) if pd.notna(row.get('Scoring Methodology Notes', '')) else None,
                            'readiness_status': str(row.get('Readiness Status (Draft / Needs Data / Complete)', '')) if pd.notna(row.get('Readiness Status (Draft / Needs Data / Complete)', '')) else None,
                            'linked_sheet_or_tab': str(row.get('Linked Sheet or Tab', '')) if pd.notna(row.get('Linked Sheet or Tab', '')) else None,
                            'greenwashing_trust': str(row.get('Greenwashing Trust', '')) if pd.notna(row.get('Greenwashing Trust', '')) else None,
                            'variable_data_fields_required': str(row.get('Variable Data Fields required', '')) if pd.notna(row.get('Variable Data Fields required', '')) else None
                        }
                    }
                    metrics.append(metric_data)
            
            print(f"Processed {len(metrics)} metrics from {sheet_name}")
            return metrics
            
        except Exception as e:
            print(f"Error processing sheet {sheet_name}: {e}")
            return []
    
    def populate_database(self):
        """Populate the database with processed data."""
        print("Starting database population...")
        
        # Process data sources
        data_sources = self.process_data_sources_sheet()
        
        # Insert data sources
        for source_data in data_sources:
            try:
                source_create = DataSourceCreate(**source_data)
                self.crud.create_data_source(source_create)
                print(f"Created data source: {source_data['name']}")
            except Exception as e:
                print(f"Error creating data source {source_data['name']}: {e}")
        
        # Process all metric sheets
        all_metrics = []
        metric_sheets = [
            '1 Necessity Score (Core)',
            '2 Resource Extraction & Scarcit',
            '3 Artificial Support Score (Cor',
            '4 Emissions Score',
            '5 Ecological Score',
            '8 Workforce Transition Score',
            '9 Infrastructure Repurposing Sc',
            '11 Monopoly & Corporate Control',
            '20 Economic Score',
            '24.  Technological Disruption S'
        ]
        
        for sheet_name in metric_sheets:
            metrics = self.process_metric_sheet(sheet_name)
            all_metrics.extend(metrics)
        
        # Insert metric definitions
        for metric_data in all_metrics:
            try:
                metric_create = MetricDefinitionCreate(**metric_data)
                self.crud.create_metric_definition(metric_create)
                print(f"Created metric: {metric_data['name']}")
            except Exception as e:
                print(f"Error creating metric {metric_data['name']}: {e}")
        
        print(f"Database population complete. Inserted {len(data_sources)} data sources and {len(all_metrics)} metrics.")

def main():
    excel_file = "/workspaces/FVI-ANSHU/FVI Scoring Metrics_Coal.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"Excel file not found: {excel_file}")
        return
    
    processor = FVIDataProcessor(excel_file)
    processor.populate_database()

if __name__ == "__main__":
    main()
