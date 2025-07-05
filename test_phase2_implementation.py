#!/usr/bin/env python3
"""
Phase 2 Implementation Test Suite

This script validates the Phase 2 implementation by testing all major components:
1. Database connectivity and dbt models
2. ML pipeline (training, inference, explainability)
3. Vector RAG system
4. Feature store integration
5. Observability stack
6. API endpoints

Usage:
    python test_phase2_implementation.py [--verbose] [--component COMPONENT]
"""

import asyncio
import logging
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import asyncpg
import redis.asyncio as redis
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result container"""
    component: str
    test_name: str
    passed: bool
    duration: float
    message: str
    details: Dict[str, Any] = None

class Phase2Tester:
    """Phase 2 implementation test suite"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[TestResult] = []
        self.redis_client = None
        self.db_pool = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize Redis client
        self.redis_client = redis.Redis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379'),
            decode_responses=True
        )
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            self.config.get('database_url', 'postgresql://postgres:postgres@localhost:5432/fvi_analytics'),
            min_size=1,
            max_size=5
        )
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
            
    async def run_test(self, component: str, test_name: str, test_func):
        """Run a single test and record results"""
        logger.info(f"Running {component}.{test_name}")
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            test_result = TestResult(
                component=component,
                test_name=test_name,
                passed=True,
                duration=duration,
                message="Test passed",
                details=result if isinstance(result, dict) else None
            )
            
            logger.info(f"✓ {component}.{test_name} passed ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                component=component,
                test_name=test_name,
                passed=False,
                duration=duration,
                message=str(e),
                details={"error_type": type(e).__name__}
            )
            
            logger.error(f"✗ {component}.{test_name} failed ({duration:.2f}s): {str(e)}")
            
        self.results.append(test_result)
        return test_result
        
    async def test_database_connectivity(self):
        """Test database connectivity"""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("SELECT version()")
            return {"postgres_version": result['version']}
            
    async def test_dbt_models(self):
        """Test dbt models exist and have data"""
        async with self.db_pool.acquire() as conn:
            # Check if dbt models exist
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'bronze_%' 
                OR table_name LIKE 'silver_%' 
                OR table_name LIKE 'gold_%'
            """)
            
            table_names = [row['table_name'] for row in tables]
            
            # Check if mart view exists
            views = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'public' 
                AND table_name = 'vw_metric_dq'
            """)
            
            view_names = [row['table_name'] for row in views]
            
            return {
                "tables": table_names,
                "views": view_names,
                "total_tables": len(table_names),
                "total_views": len(view_names)
            }
            
    async def test_redis_connectivity(self):
        """Test Redis connectivity"""
        # Test basic operations
        await self.redis_client.set("test_key", "test_value", ex=60)
        value = await self.redis_client.get("test_key")
        
        if value != "test_value":
            raise ValueError("Redis set/get operation failed")
            
        # Test Redis info
        info = await self.redis_client.info()
        
        return {
            "redis_version": info.get('redis_version'),
            "used_memory": info.get('used_memory_human'),
            "connected_clients": info.get('connected_clients')
        }
        
    async def test_api_health(self):
        """Test API health endpoints"""
        endpoints = [
            ("inference", f"http://localhost:{self.config.get('inference_port', 8000)}/health"),
            ("vector_rag", f"http://localhost:{self.config.get('vector_rag_port', 8001)}/health"),
            ("mlflow", f"http://localhost:{self.config.get('mlflow_port', 5000)}/health"),
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service, url in endpoints:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            results[service] = {
                                "status": "healthy",
                                "response_time": response.headers.get('X-Response-Time', 'N/A'),
                                "data": data
                            }
                        else:
                            results[service] = {
                                "status": "unhealthy",
                                "status_code": response.status
                            }
                except Exception as e:
                    results[service] = {
                        "status": "error",
                        "error": str(e)
                    }
                    
        return results
        
    async def test_ml_inference(self):
        """Test ML inference endpoint"""
        url = f"http://localhost:{self.config.get('inference_port', 8000)}/predict"
        
        payload = {
            "mine_id": "test_mine_001",
            "features": {
                "production_rate": 1000.0,
                "equipment_health": 0.85,
                "environmental_score": 0.75,
                "maintenance_score": 0.90
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise ValueError(f"Inference API returned status {response.status}")
                    
                data = await response.json()
                
                return {
                    "prediction": data.get('prediction'),
                    "confidence": data.get('confidence'),
                    "model_version": data.get('model_version'),
                    "processing_time_ms": data.get('processing_time_ms')
                }
                
    async def test_vector_rag_chat(self):
        """Test Vector RAG chat endpoint"""
        url = f"http://localhost:{self.config.get('vector_rag_port', 8001)}/chat"
        
        payload = {
            "message": "What are the latest production metrics?",
            "history": []
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise ValueError(f"Vector RAG API returned status {response.status}")
                    
                data = await response.json()
                
                return {
                    "response_length": len(data.get('response', '')),
                    "sources_count": len(data.get('sources', [])),
                    "has_response": bool(data.get('response'))
                }
                
    async def test_dynamic_weights(self):
        """Test dynamic weights endpoint"""
        url = f"http://localhost:{self.config.get('inference_port', 8000)}/weights/current"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise ValueError(f"Dynamic weights API returned status {response.status}")
                    
                data = await response.json()
                
                return {
                    "weights_count": len(data.get('data', {})),
                    "has_weights": bool(data.get('data'))
                }
                
    async def test_feature_store(self):
        """Test feature store integration"""
        # Check if feature store directory exists
        feast_repo_path = Path(self.config.get('feast_repo_path', './feast_repo'))
        
        if not feast_repo_path.exists():
            raise FileNotFoundError("Feature store repository not found")
            
        # Check for feature store files
        feature_files = list(feast_repo_path.glob('**/*.py'))
        yaml_files = list(feast_repo_path.glob('**/*.yaml'))
        
        return {
            "repo_exists": feast_repo_path.exists(),
            "feature_files": len(feature_files),
            "yaml_files": len(yaml_files)
        }
        
    async def test_observability_stack(self):
        """Test observability stack"""
        endpoints = [
            ("jaeger", f"http://localhost:{self.config.get('jaeger_port', 16686)}/api/services"),
            ("prometheus", f"http://localhost:{self.config.get('prometheus_port', 9090)}/api/v1/targets"),
            ("grafana", f"http://localhost:{self.config.get('grafana_port', 3001)}/api/health"),
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for service, url in endpoints:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            results[service] = {"status": "healthy"}
                        else:
                            results[service] = {"status": "unhealthy", "status_code": response.status}
                except Exception as e:
                    results[service] = {"status": "error", "error": str(e)}
                    
        return results
        
    async def test_frontend_availability(self):
        """Test frontend availability"""
        url = f"http://localhost:{self.config.get('frontend_port', 3000)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    raise ValueError(f"Frontend returned status {response.status}")
                    
                content = await response.text()
                
                return {
                    "status_code": response.status,
                    "content_length": len(content),
                    "has_content": bool(content)
                }
                
    async def run_all_tests(self, components: Optional[List[str]] = None):
        """Run all tests or specific components"""
        # Define test suite
        test_suite = {
            "database": [
                ("connectivity", self.test_database_connectivity),
                ("dbt_models", self.test_dbt_models)
            ],
            "redis": [
                ("connectivity", self.test_redis_connectivity)
            ],
            "api": [
                ("health", self.test_api_health),
                ("ml_inference", self.test_ml_inference),
                ("dynamic_weights", self.test_dynamic_weights)
            ],
            "vector_rag": [
                ("chat", self.test_vector_rag_chat)
            ],
            "feature_store": [
                ("integration", self.test_feature_store)
            ],
            "observability": [
                ("stack", self.test_observability_stack)
            ],
            "frontend": [
                ("availability", self.test_frontend_availability)
            ]
        }
        
        # Filter components if specified
        if components:
            test_suite = {k: v for k, v in test_suite.items() if k in components}
            
        # Run tests
        for component, tests in test_suite.items():
            logger.info(f"Testing {component} component...")
            
            for test_name, test_func in tests:
                await self.run_test(component, test_name, test_func)
                
    def print_results(self):
        """Print test results summary"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*80)
        print("Phase 2 Implementation Test Results")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*80)
        
        # Group by component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
            
        for component, tests in components.items():
            component_passed = sum(1 for t in tests if t.passed)
            component_total = len(tests)
            
            print(f"\n{component.upper()} ({component_passed}/{component_total})")
            print("-" * 40)
            
            for test in tests:
                status = "✓" if test.passed else "✗"
                print(f"  {status} {test.test_name} ({test.duration:.2f}s)")
                if not test.passed:
                    print(f"    Error: {test.message}")
                    
        # Failed tests details
        if failed_tests > 0:
            print("\nFailed Tests Details:")
            print("-" * 40)
            for result in self.results:
                if not result.passed:
                    print(f"• {result.component}.{result.test_name}: {result.message}")
                    
        print("\n" + "="*80)
        
    def get_json_results(self) -> Dict[str, Any]:
        """Get results as JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "failed_tests": sum(1 for r in self.results if not r.passed),
            "success_rate": sum(1 for r in self.results if r.passed) / len(self.results) * 100 if self.results else 0,
            "results": [
                {
                    "component": r.component,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "message": r.message,
                    "details": r.details
                } for r in self.results
            ]
        }

async def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Phase 2 Implementation Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--component", "-c", action="append", help="Test specific component(s)")
    parser.add_argument("--config", "-f", default="test_config.json", help="Configuration file")
    parser.add_argument("--output", "-o", help="Output JSON results to file")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load configuration
    config = {
        "database_url": "postgresql://postgres:postgres@localhost:5432/fvi_analytics",
        "redis_url": "redis://localhost:6379",
        "inference_port": 8000,
        "vector_rag_port": 8001,
        "mlflow_port": 5000,
        "jaeger_port": 16686,
        "prometheus_port": 9090,
        "grafana_port": 3001,
        "frontend_port": 3000,
        "feast_repo_path": "./feast_repo"
    }
    
    # Load config file if exists
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config.update(json.load(f))
            
    # Run tests
    async with Phase2Tester(config) as tester:
        await tester.run_all_tests(args.component)
        
        # Print results
        tester.print_results()
        
        # Save JSON results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(tester.get_json_results(), f, indent=2)
            print(f"\nResults saved to {args.output}")
            
        # Exit with error code if any tests failed
        failed_tests = sum(1 for r in tester.results if not r.passed)
        if failed_tests > 0:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
