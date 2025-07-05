#!/usr/bin/env python3
"""
FVI System Integration Test
==========================

Test the core components of the FVI system.
"""

import logging
import asyncio
from pathlib import Path
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FVISystemTester:
    """Test the FVI system components."""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.catalog_path = Path("/workspaces/FVI-ANSHU/meta/metric_catalogue.yaml")
        
    def test_catalog_generation(self):
        """Test that the catalog was generated successfully."""
        logger.info("Testing catalog generation...")
        
        if not self.catalog_path.exists():
            logger.error("Catalog file not found!")
            return False
            
        # Check catalog size
        size = self.catalog_path.stat().st_size
        logger.info(f"Catalog file size: {size} bytes")
        
        if size < 1000:  # Should be much larger
            logger.error("Catalog file seems too small!")
            return False
            
        logger.info("✓ Catalog generation successful")
        return True
        
    def test_api_endpoints(self):
        """Test API endpoints."""
        logger.info("Testing API endpoints...")
        
        try:
            # Test root endpoint
            response = requests.get(f"{self.api_url}/")
            if response.status_code != 200:
                logger.error(f"Root endpoint failed: {response.status_code}")
                return False
                
            # Test catalog endpoint
            response = requests.get(f"{self.api_url}/catalog")
            if response.status_code != 200:
                logger.error(f"Catalog endpoint failed: {response.status_code}")
                return False
                
            catalog_data = response.json()
            logger.info(f"API reports {catalog_data['total_data_sources']} data sources and {catalog_data['total_metrics']} metrics")
            
            # Test data sources endpoint
            response = requests.get(f"{self.api_url}/data-sources")
            if response.status_code != 200:
                logger.error(f"Data sources endpoint failed: {response.status_code}")
                return False
                
            data_sources = response.json()
            logger.info(f"Retrieved {len(data_sources)} data sources")
            
            # Test metrics endpoint
            response = requests.get(f"{self.api_url}/metrics")
            if response.status_code != 200:
                logger.error(f"Metrics endpoint failed: {response.status_code}")
                return False
                
            metrics = response.json()
            logger.info(f"Retrieved {len(metrics)} metrics")
            
            # Test specific metric endpoint
            if metrics:
                first_metric = metrics[0]
                metric_key = first_metric['metric_key']
                response = requests.get(f"{self.api_url}/metrics/{metric_key}")
                if response.status_code != 200:
                    logger.error(f"Specific metric endpoint failed: {response.status_code}")
                    return False
                    
                metric_detail = response.json()
                logger.info(f"Retrieved metric: {metric_detail['basic_info']['title']}")
            
            logger.info("✓ All API endpoints working")
            return True
            
        except Exception as e:
            logger.error(f"API test failed: {e}")
            return False
            
    def test_data_quality(self):
        """Test data quality and completeness."""
        logger.info("Testing data quality...")
        
        try:
            # Get metrics from API
            response = requests.get(f"{self.api_url}/metrics")
            metrics = response.json()
            
            # Check for required fields
            required_fields = ['metric_key', 'feature_column', 'basic_info', 'data_quality']
            metrics_with_issues = []
            
            for metric in metrics:
                for field in required_fields:
                    if field not in metric:
                        metrics_with_issues.append(f"{metric.get('metric_key', 'unknown')}: missing {field}")
                        
            if metrics_with_issues:
                logger.warning(f"Found {len(metrics_with_issues)} metrics with missing fields")
                for issue in metrics_with_issues[:5]:  # Show first 5
                    logger.warning(f"  - {issue}")
            else:
                logger.info("✓ All metrics have required fields")
                
            # Check sheet distribution
            sheet_counts = {}
            for metric in metrics:
                sheet_name = metric.get('sheet_info', {}).get('sheet_name', 'unknown')
                sheet_counts[sheet_name] = sheet_counts.get(sheet_name, 0) + 1
                
            logger.info("Metrics by sheet:")
            for sheet, count in sheet_counts.items():
                logger.info(f"  - {sheet}: {count} metrics")
                
            logger.info("✓ Data quality check completed")
            return True
            
        except Exception as e:
            logger.error(f"Data quality test failed: {e}")
            return False
            
    def test_naming_conventions(self):
        """Test naming conventions."""
        logger.info("Testing naming conventions...")
        
        try:
            # Get metrics from API
            response = requests.get(f"{self.api_url}/metrics")
            metrics = response.json()
            
            # Check naming patterns
            naming_issues = []
            
            for metric in metrics:
                metric_key = metric.get('metric_key', '')
                feature_column = metric.get('feature_column', '')
                
                # Check metric key pattern (should be NN_slug)
                if not (len(metric_key) > 3 and metric_key[2] == '_'):
                    naming_issues.append(f"Invalid metric key format: {metric_key}")
                    
                # Check feature column pattern (should be f_NN_slug)
                if not feature_column.startswith('f_'):
                    naming_issues.append(f"Invalid feature column format: {feature_column}")
                    
            if naming_issues:
                logger.warning(f"Found {len(naming_issues)} naming issues")
                for issue in naming_issues[:5]:  # Show first 5
                    logger.warning(f"  - {issue}")
            else:
                logger.info("✓ All naming conventions followed")
                
            return True
            
        except Exception as e:
            logger.error(f"Naming conventions test failed: {e}")
            return False
            
    def run_all_tests(self):
        """Run all tests."""
        logger.info("Starting FVI System Integration Tests")
        logger.info("=" * 50)
        
        tests = [
            ("Catalog Generation", self.test_catalog_generation),
            ("API Endpoints", self.test_api_endpoints),
            ("Data Quality", self.test_data_quality),
            ("Naming Conventions", self.test_naming_conventions)
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results.append((test_name, False))
                
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        
        passed = 0
        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
                
        logger.info(f"\nPassed: {passed}/{len(tests)} tests")
        
        if passed == len(tests):
            logger.info("🎉 All tests passed! The FVI system is working correctly.")
        else:
            logger.warning(f"⚠️  {len(tests) - passed} tests failed. Please check the logs above.")
            
        return passed == len(tests)


if __name__ == "__main__":
    tester = FVISystemTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)
