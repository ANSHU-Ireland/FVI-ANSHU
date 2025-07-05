"""
OpenTelemetry instrumentation for FVI Analytics Platform
Provides distributed tracing, metrics, and observability across all services.
"""

import os
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import logging
import functools
import time
from typing import Any, Callable, Dict, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FVITelemetry:
    """
    Central telemetry configuration for FVI Analytics Platform
    """
    
    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
        self.resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
        })
        
        # Initialize providers
        self._setup_tracing()
        self._setup_metrics()
        self._setup_propagation()
        self._setup_instrumentation()
        
    def _setup_tracing(self):
        """Setup distributed tracing with Jaeger"""
        # Create tracer provider
        trace.set_tracer_provider(TracerProvider(resource=self.resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
            collector_endpoint=os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
    def _setup_metrics(self):
        """Setup metrics with Prometheus"""
        # Create metric reader
        metric_reader = PrometheusMetricReader()
        
        # Create meter provider
        metrics.set_meter_provider(MeterProvider(
            resource=self.resource,
            metric_readers=[metric_reader]
        ))
        
        # Get meter
        self.meter = metrics.get_meter(__name__)
        
        # Create common metrics
        self.request_counter = self.meter.create_counter(
            "fvi_requests_total",
            description="Total number of requests processed",
        )
        
        self.request_duration = self.meter.create_histogram(
            "fvi_request_duration_seconds",
            description="Request duration in seconds",
            unit="s",
        )
        
        self.error_counter = self.meter.create_counter(
            "fvi_errors_total",
            description="Total number of errors",
        )
        
        self.prediction_counter = self.meter.create_counter(
            "fvi_predictions_total",
            description="Total number of ML predictions",
        )
        
        self.weight_update_counter = self.meter.create_counter(
            "fvi_weight_updates_total",
            description="Total number of weight updates",
        )
        
        self.data_quality_gauge = self.meter.create_up_down_counter(
            "fvi_data_quality_score",
            description="Current data quality score",
        )
        
    def _setup_propagation(self):
        """Setup trace context propagation"""
        set_global_textmap(TraceContextTextMapPropagator())
        
    def _setup_instrumentation(self):
        """Setup automatic instrumentation"""
        # HTTP requests
        RequestsInstrumentor().instrument()
        
        # Database
        Psycopg2Instrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        # Redis
        RedisInstrumentor().instrument()
        
        # Logging
        LoggingInstrumentor().instrument()
        
    def instrument_fastapi(self, app):
        """Instrument FastAPI application"""
        FastAPIInstrumentor.instrument_app(app)
        
    def trace_function(self, name: str = None):
        """Decorator to trace function calls"""
        def decorator(func: Callable) -> Callable:
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name) as span:
                    try:
                        # Add function attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        # Execute function
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        
                        # Record metrics
                        duration = time.time() - start_time
                        self.request_duration.record(duration, {
                            "function": func.__name__,
                            "status": "success"
                        })
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        self.error_counter.add(1, {
                            "function": func.__name__,
                            "error_type": type(e).__name__
                        })
                        raise
                        
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name) as span:
                    try:
                        # Add function attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        # Execute function
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Record metrics
                        duration = time.time() - start_time
                        self.request_duration.record(duration, {
                            "function": func.__name__,
                            "status": "success"
                        })
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        self.error_counter.add(1, {
                            "function": func.__name__,
                            "error_type": type(e).__name__
                        })
                        raise
                        
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    def trace_ml_prediction(self, model_name: str, model_version: str):
        """Decorator to trace ML predictions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("ml_prediction") as span:
                    try:
                        # Add ML-specific attributes
                        span.set_attribute("ml.model.name", model_name)
                        span.set_attribute("ml.model.version", model_version)
                        span.set_attribute("ml.framework", "lightgbm")
                        
                        # Execute prediction
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Record metrics
                        duration = time.time() - start_time
                        self.prediction_counter.add(1, {
                            "model_name": model_name,
                            "model_version": model_version
                        })
                        
                        self.request_duration.record(duration, {
                            "operation": "ml_prediction",
                            "model": model_name
                        })
                        
                        # Add result attributes
                        if hasattr(result, 'prediction'):
                            span.set_attribute("ml.prediction.value", float(result.prediction))
                        if hasattr(result, 'confidence'):
                            span.set_attribute("ml.prediction.confidence", float(result.confidence))
                            
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        self.error_counter.add(1, {
                            "operation": "ml_prediction",
                            "model": model_name,
                            "error_type": type(e).__name__
                        })
                        raise
                        
            return wrapper
        return decorator
        
    def trace_weight_update(self, engine_type: str = "information_gain"):
        """Decorator to trace weight updates"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("weight_update") as span:
                    try:
                        # Add weight update attributes
                        span.set_attribute("weight.engine.type", engine_type)
                        
                        # Execute update
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Record metrics
                        duration = time.time() - start_time
                        self.weight_update_counter.add(1, {
                            "engine_type": engine_type
                        })
                        
                        self.request_duration.record(duration, {
                            "operation": "weight_update",
                            "engine": engine_type
                        })
                        
                        # Add result attributes
                        if isinstance(result, dict):
                            span.set_attribute("weight.features.count", len(result))
                            
                        return result
                        
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        self.error_counter.add(1, {
                            "operation": "weight_update",
                            "engine": engine_type,
                            "error_type": type(e).__name__
                        })
                        raise
                        
            return wrapper
        return decorator
        
    def record_data_quality(self, score: float, dimension: str = "overall"):
        """Record data quality metrics"""
        self.data_quality_gauge.add(score, {
            "dimension": dimension
        })
        
    def create_child_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a child span with optional attributes"""
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span
        
    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID"""
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().trace_id, '032x')
        return None
        
    def get_span_id(self) -> Optional[str]:
        """Get current span ID"""
        span = trace.get_current_span()
        if span:
            return format(span.get_span_context().span_id, '016x')
        return None

# Global telemetry instance
_telemetry_instance = None

def get_telemetry(service_name: str = None, service_version: str = "1.0.0") -> FVITelemetry:
    """Get or create global telemetry instance"""
    global _telemetry_instance
    
    if _telemetry_instance is None:
        if service_name is None:
            service_name = os.getenv("SERVICE_NAME", "fvi-analytics")
        _telemetry_instance = FVITelemetry(service_name, service_version)
        
    return _telemetry_instance

def trace_function(name: str = None):
    """Convenience decorator for tracing functions"""
    return get_telemetry().trace_function(name)

def trace_ml_prediction(model_name: str, model_version: str):
    """Convenience decorator for tracing ML predictions"""
    return get_telemetry().trace_ml_prediction(model_name, model_version)

def trace_weight_update(engine_type: str = "information_gain"):
    """Convenience decorator for tracing weight updates"""
    return get_telemetry().trace_weight_update(engine_type)

# Context manager for manual span creation
class traced_operation:
    """Context manager for manual span creation"""
    
    def __init__(self, name: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.attributes = attributes or {}
        self.telemetry = get_telemetry()
        self.span = None
        
    def __enter__(self):
        self.span = self.telemetry.create_child_span(self.name, self.attributes)
        return self.span
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            self.span.end()
            
# Example usage
if __name__ == "__main__":
    # Initialize telemetry
    telemetry = get_telemetry("fvi-example")
    
    # Example traced function
    @trace_function("example_function")
    def example_function():
        print("This function is traced!")
        
    # Example ML prediction
    @trace_ml_prediction("lightgbm", "v2.1.0")
    def example_prediction():
        return {"prediction": 0.85, "confidence": 0.92}
        
    # Example weight update
    @trace_weight_update("information_gain")
    def example_weight_update():
        return {"feature_1": 0.3, "feature_2": 0.7}
        
    # Manual span creation
    with traced_operation("manual_operation", {"key": "value"}):
        print("This operation is manually traced!")
        
    # Test functions
    example_function()
    result = example_prediction()
    weights = example_weight_update()
    
    print(f"Prediction: {result}")
    print(f"Weights: {weights}")
    print(f"Current trace ID: {telemetry.get_trace_id()}")
