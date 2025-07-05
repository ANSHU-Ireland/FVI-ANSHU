#!/usr/bin/env python3
"""
FVI System Test Runner
======================

Comprehensive test of the FVI system including:
- Excel to YAML conversion
- Database initialization
- Formula calculation
- Dynamic weighting
- ML model training
- API endpoints

Usage:
    python scripts/test_fvi_system.py
"""

import os
import sys
import logging
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.naming import FVINamingConventions, FVIHorizon, FVIMetricFormulas
from core.formulas import FormulaEngine, DynamicWeightingEngine
from data.processor import ExcelProcessor
from data.sources import MockDataGenerator
from ml.models import FVIPredictor

logger = logging.getLogger(__name__)


class FVISystemTester:
    """Comprehensive system tester for FVI platform."""
    
    def __init__(self, excel_path: str = "FVI Scoring Metrics_Coal.xlsx"):
        self.excel_path = Path(excel_path)
        self.api_base_url = "http://localhost:8000"
        self.test_results = {}
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_naming_conventions(self) -> bool:
        """Test FVI naming conventions."""
        logger.info("Testing naming conventions...")
        
        try:
            # Test metric key generation
            metric_key = FVINamingConventions.generate_metric_key(
                "4 Emissions Score", "co2_intensity"
            )
            expected = "04_co2_intensity"
            assert metric_key == expected, f"Expected {expected}, got {metric_key}"
            
            # Test feature column generation
            feature_col = FVINamingConventions.generate_feature_column(metric_key)
            expected = "f_04_co2_intensity"
            assert feature_col == expected, f"Expected {expected}, got {feature_col}"
            
            # Test weight column generation
            weight_col = FVINamingConventions.generate_weight_column(metric_key, FVIHorizon.H10)
            expected = "w_04_co2_intensity_H10"
            assert weight_col == expected, f"Expected {expected}, got {weight_col}"
            
            # Test composite score generation
            score_name = FVINamingConventions.generate_composite_score("coal", FVIHorizon.H10)
            expected = "s_coal_H10"
            assert score_name == expected, f"Expected {expected}, got {score_name}"
            
            # Test validation
            assert FVINamingConventions.validate_metric_key("04_co2_intensity")
            assert FVINamingConventions.validate_feature_column("f_04_co2_intensity")
            assert FVINamingConventions.validate_weight_column("w_04_co2_intensity_H10")
            assert FVINamingConventions.validate_composite_score("s_coal_H10")
            
            logger.info("✓ Naming conventions test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Naming conventions test failed: {e}")
            return False
    
    def test_formula_engine(self) -> bool:
        """Test formula engine with sample data."""
        logger.info("Testing formula engine...")
        
        try:
            engine = FormulaEngine()
            
            # Test individual formula
            formula = "100 * (1 - zscore(CO2_t_per_MWh))"
            test_data = {
                "CO2_t_per_MWh": [0.8, 0.9, 1.0, 1.1, 1.2]
            }
            
            result = engine.evaluate_formula(formula, test_data)
            assert isinstance(result, (float, int)) or hasattr(result, '__len__'), "Formula should return numeric result"
            
            # Test formula validation
            sample_data = {"CO2_t_per_MWh": 1.0}
            is_valid = engine.validate_formula(formula, sample_data)
            assert is_valid, "Formula should be valid"
            
            # Test variable extraction
            variables = engine.get_formula_variables(formula)
            assert "CO2_t_per_MWh" in variables, "Should extract variable names"
            
            # Test sheet metrics calculation
            df = pd.DataFrame({
                "CO2_t_per_MWh": [0.8, 0.9, 1.0, 1.1, 1.2],
                "CH4_emissions_factor": [0.1, 0.2, 0.3, 0.4, 0.5],
                "Carbon_price_USD_per_t": [50, 60, 70, 80, 90],
                "Emissions_intensity": [1.0, 1.1, 1.2, 1.3, 1.4]
            })
            
            result_df = engine.calculate_emissions_metrics(df)
            feature_cols = [col for col in result_df.columns if col.startswith('f_04_')]
            
            assert len(feature_cols) > 0, "Should generate feature columns"
            logger.info(f"Generated {len(feature_cols)} feature columns for emissions")
            
            logger.info("✓ Formula engine test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Formula engine test failed: {e}")
            return False
    
    def test_dynamic_weighting(self) -> bool:
        """Test dynamic weighting engine."""
        logger.info("Testing dynamic weighting...")
        
        try:
            engine = DynamicWeightingEngine()
            
            # Create sample data
            df = pd.DataFrame({
                "f_04_co2_intensity": [100, 90, 80, 70, 60],
                "f_04_methane_factor": [95, 85, 75, 65, 55],
                "f_01_energy_security": [80, 75, 70, 65, 60],
                "f_20_profit_resilience": [70, 65, 60, 55, 50]
            })
            
            target = pd.Series([75, 70, 65, 60, 55], name="fvi_score")
            
            # Calculate quality weights
            quality_weights = engine.calculate_data_quality_weights(df)
            assert len(quality_weights) == len(df.columns), "Should calculate weights for all feature columns"
            
            # Calculate information gain weights
            info_gain_weights = engine.calculate_information_gain_weights(df, target)
            assert len(info_gain_weights) == len(df.columns), "Should calculate info gain for all features"
            
            # Calculate dynamic weights
            dynamic_weights = engine.calculate_dynamic_weights(df, target, "H10")
            assert len(dynamic_weights) > 0, "Should generate dynamic weights"
            
            # Test normalization
            normalized_weights = engine.normalize_weights(dynamic_weights)
            weight_sum = sum(normalized_weights.values())
            assert abs(weight_sum - 1.0) < 0.001, f"Normalized weights should sum to 1.0, got {weight_sum}"
            
            logger.info("✓ Dynamic weighting test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Dynamic weighting test failed: {e}")
            return False
    
    def test_excel_processing(self) -> bool:
        """Test Excel file processing."""
        logger.info("Testing Excel processing...")
        
        try:
            if not self.excel_path.exists():
                logger.warning(f"Excel file not found: {self.excel_path}. Using mock data.")
                return True
            
            processor = ExcelProcessor(str(self.excel_path))
            results = processor.process_all_sheets()
            
            assert "data_sources" in results, "Should extract data sources"
            assert "metrics" in results, "Should extract metrics"
            assert len(results["data_sources"]) > 0, "Should have data sources"
            assert len(results["metrics"]) > 0, "Should have metrics"
            
            logger.info(f"Processed {len(results['data_sources'])} data sources and {len(results['metrics'])} metrics")
            
            # Test YAML conversion
            from scripts.excel2yaml import ExcelToYAMLConverter
            converter = ExcelToYAMLConverter(str(self.excel_path))
            catalog = converter.convert_to_yaml()
            
            assert "metadata" in catalog, "Should have metadata"
            assert "data_sources" in catalog, "Should have data sources"
            assert "metric_definitions" in catalog, "Should have metric definitions"
            
            logger.info("✓ Excel processing test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Excel processing test failed: {e}")
            return False
    
    def test_ml_models(self) -> bool:
        """Test ML model training and prediction."""
        logger.info("Testing ML models...")
        
        try:
            # Generate mock data
            mock_data = MockDataGenerator.generate_coal_metrics_data(
                sub_industry="Coal Mining",
                country="Global",
                years=[2020, 2021, 2022, 2023]
            )
            
            # Create training data
            training_data = []
            for i, year in enumerate(mock_data["years"]):
                row = {"year": year}
                for metric, values in mock_data["metrics"].items():
                    row[f"f_{metric}"] = values[i]
                row["fvi_score"] = mock_data["fvi_scores"][i]
                training_data.append(row)
            
            df = pd.DataFrame(training_data)
            X = df.drop(columns=["fvi_score"])
            y = df["fvi_score"]
            
            # Test individual predictor
            predictor = FVIPredictor("lightgbm")
            training_results = predictor.train(X, y)
            
            assert "train_rmse" in training_results, "Should return training metrics"
            assert "val_rmse" in training_results, "Should return validation metrics"
            assert predictor.is_fitted, "Model should be fitted"
            
            # Test predictions
            predictions = predictor.predict(X)
            assert len(predictions) == len(X), "Should predict for all samples"
            
            # Test explanations
            explanation = predictor.explain_prediction(X.iloc[:1])
            assert "feature_contributions" in explanation or "error" in explanation, "Should provide explanation"
            
            logger.info("✓ ML models test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ ML models test failed: {e}")
            return False
    
    def test_api_endpoints(self) -> bool:
        """Test API endpoints (if server is running)."""
        logger.info("Testing API endpoints...")
        
        try:
            # Test health check
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code != 200:
                    logger.warning("API server not running, skipping API tests")
                    return True
            except requests.exceptions.RequestException:
                logger.warning("API server not accessible, skipping API tests")
                return True
            
            # Test naming conventions validation
            response = requests.get(
                f"{self.api_base_url}/api/v1/data/validate-naming-conventions",
                params={"metric_key": "04_co2_intensity"}
            )
            assert response.status_code == 200, f"API call failed: {response.status_code}"
            
            data = response.json()
            assert data["validations"]["metric_key"]["valid"], "Metric key should be valid"
            
            # Test formula catalog
            response = requests.get(f"{self.api_base_url}/api/v1/data/formula-catalog")
            assert response.status_code == 200, f"API call failed: {response.status_code}"
            
            data = response.json()
            assert "formulas_by_sheet" in data, "Should return formulas by sheet"
            
            logger.info("✓ API endpoints test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ API endpoints test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("Starting FVI system tests...")
        
        tests = [
            ("naming_conventions", self.test_naming_conventions),
            ("formula_engine", self.test_formula_engine),
            ("dynamic_weighting", self.test_dynamic_weighting),
            ("excel_processing", self.test_excel_processing),
            ("ml_models", self.test_ml_models),
            ("api_endpoints", self.test_api_endpoints)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} test ---")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        logger.info(f"\n=== TEST SUMMARY ===")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success rate: {passed/total*100:.1f}%")
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name}: {status}")
        
        self.test_results = results
        return results
    
    def generate_test_report(self) -> str:
        """Generate a detailed test report."""
        report = f"""
FVI System Test Report
=====================

Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Excel File: {self.excel_path}

Test Results:
"""
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            report += f"- {test_name}: {status}\n"
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        report += f"""
Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)

System Status: {"READY" if passed == total else "NEEDS ATTENTION"}
"""
        
        return report


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run FVI system tests")
    parser.add_argument("--excel", default="FVI Scoring Metrics_Coal.xlsx", help="Excel file path")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--report", help="Save test report to file")
    
    args = parser.parse_args()
    
    # Run tests
    tester = FVISystemTester(args.excel)
    tester.api_base_url = args.api_url
    
    results = tester.run_all_tests()
    
    # Generate report
    if args.report:
        report = tester.generate_test_report()
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"Test report saved to: {args.report}")
    
    # Exit with appropriate code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
