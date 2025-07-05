# FVI Analytics Platform - Complete Setup and Usage Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Phase 2 Components](#phase-2-components)
6. [Running the System](#running-the-system)
7. [Testing and Validation](#testing-and-validation)
8. [API Documentation](#api-documentation)
9. [Troubleshooting](#troubleshooting)
10. [Development Guide](#development-guide)
11. [Deployment](#deployment)
12. [Monitoring](#monitoring)

## 🎯 Overview

The FVI Analytics Platform is a comprehensive, production-grade analytics solution for mining operations. Phase 2 provides:

- **Real-time Data Processing**: Multi-source data ingestion with quality scoring
- **Advanced ML Pipeline**: Dynamic weights, ONNX inference, and explainable AI
- **Vector-RAG System**: OpenAI-powered chat with document retrieval
- **Modern Frontend**: Next.js 14 dashboard with real-time updates
- **Full Observability**: OpenTelemetry tracing and monitoring

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js 14)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐│
│  │ Dashboard   │ │ Chat        │ │ Weights     │ │ Monitoring ││
│  │ (Real-time) │ │ (Streaming) │ │ (Dynamic)   │ │ (Health)   ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘│
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/WebSocket/SSE
┌─────────────────────────┴───────────────────────────────────────┐
│                    API Gateway (FastAPI)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐│
│  │ REST APIs   │ │ WebSocket   │ │ Health      │ │ Metrics    ││
│  │ (CRUD)      │ │ (Real-time) │ │ (Status)    │ │ (Telemetry)││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘│
└─────────────────────────┬───────────────────────────────────────┘
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
┌────▼────┐         ┌────▼────┐         ┌────▼────┐
│ ML      │         │Vector   │         │ Data    │
│ Services│         │ RAG     │         │ Sources │
└────┬────┘         └────┬────┘         └────┬────┘
     │                   │                   │
┌────▼────┐         ┌────▼────┐         ┌────▼────┐
│Training │         │pgvector │         │Structured│
│Inference│         │OpenAI   │         │Feeds    │
│Weights  │         │Embedding│         │APIs     │
│SHAP     │         └─────────┘         └─────────┘
└────┬────┘                                   │
     │                                        │
┌────▼────┐                             ┌────▼────┐
│ Redis   │                             │dbt      │
│ Cache   │                             │Models   │
│ Features│                             │ETL      │
└─────────┘                             └────┬────┘
                                             │
                                       ┌────▼────┐
                                       │PostgreSQL│
                                       │+pgvector │
                                       │Data WH   │
                                       └─────────┘
```

## 🖥️ System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 50 GB | 100+ GB SSD |
| Network | 10 Mbps | 100+ Mbps |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Operating System** | | |
| Linux | Ubuntu 20.04+ | Primary support |
| macOS | 11+ | Development |
| Windows | 10+ with WSL2 | Development |
| **Container Runtime** | | |
| Docker | 20.10+ | Container orchestration |
| Docker Compose | 2.0+ | Multi-service deployment |
| **Development** | | |
| Python | 3.10+ | Backend services |
| Node.js | 18+ | Frontend development |
| Git | 2.30+ | Version control |

## 🚀 Installation Guide

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/fvi-analytics.git
cd fvi-analytics

# Verify structure
ls -la
```

Expected structure:
```
fvi-analytics/
├── frontend/                 # Next.js 14 frontend
├── src/                     # Python backend services
├── dbt/                     # Data transformation models
├── docker-compose.phase2.yml # Complete orchestration
├── requirements-phase2.txt   # Python dependencies
├── .env.example             # Environment template
└── README.md               # This file
```

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (IMPORTANT - see configuration section)
nano .env
```

**Critical Environment Variables:**

```bash
# Database (Required)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/fvi_analytics
POSTGRES_PASSWORD=your_secure_password_here

# Redis (Required)
REDIS_URL=redis://localhost:6379

# OpenAI (Required for Vector RAG)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Data Source APIs (Optional but recommended)
IEA_API_KEY=your-iea-api-key
EIA_API_KEY=your-eia-api-key
UN_COMTRADE_API_KEY=your-comtrade-key

# Security (Required for production)
SECRET_KEY=your-super-secret-key-min-32-chars
```

### Step 3: Docker Infrastructure Setup

```bash
# Verify Docker is running
docker --version
docker-compose --version

# Pull required images (optional - speeds up first start)
docker-compose -f docker-compose.phase2.yml pull

# Start infrastructure services first
docker-compose -f docker-compose.phase2.yml up -d postgres redis

# Wait for services to be ready (important!)
echo "Waiting for PostgreSQL..."
until docker-compose -f docker-compose.phase2.yml exec postgres pg_isready -U postgres; do
  sleep 2
done

echo "Waiting for Redis..."
until docker-compose -f docker-compose.phase2.yml exec redis redis-cli ping; do
  sleep 2
done

echo "Infrastructure ready!"
```

### Step 4: Database Initialization

```bash
# Start dbt service to run models
docker-compose -f docker-compose.phase2.yml up -d dbt-scheduler

# Wait for dbt to complete initial run
docker-compose -f docker-compose.phase2.yml logs -f dbt-scheduler

# Verify tables were created
docker-compose -f docker-compose.phase2.yml exec postgres psql -U postgres -d fvi_analytics -c "\dt"
```

Expected output should show tables like:
```
bronze_structured_feeds
silver_mine_metrics
gold_mine_features
vw_metric_dq
```

### Step 5: ML Services

```bash
# Start MLflow tracking server
docker-compose -f docker-compose.phase2.yml up -d mlflow

# Start inference service
docker-compose -f docker-compose.phase2.yml up -d inference-service

# Start vector RAG service
docker-compose -f docker-compose.phase2.yml up -d vector-rag-service

# Start dynamic weights engine
docker-compose -f docker-compose.phase2.yml up -d weights-scheduler

# Verify all services are healthy
docker-compose -f docker-compose.phase2.yml ps
```

### Step 6: Monitoring Stack

```bash
# Start observability services
docker-compose -f docker-compose.phase2.yml up -d prometheus grafana jaeger

# Wait for services to be ready
sleep 30

# Verify monitoring stack
curl http://localhost:9090/api/v1/targets  # Prometheus
curl http://localhost:16686/api/services   # Jaeger
curl http://localhost:3001/api/health      # Grafana
```

### Step 7: Frontend Development Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list --depth=0

# Start development server
npm run dev
```

The frontend will be available at: http://localhost:3000

### Step 8: Verification

```bash
# Run comprehensive test suite
python test_phase2_implementation.py

# Check individual service health
curl http://localhost:8000/health     # Inference API
curl http://localhost:8001/health     # Vector RAG
curl http://localhost:5000/health     # MLflow
```

## ⚙️ Configuration

### Database Configuration

```bash
# PostgreSQL settings in .env
DATABASE_URL=postgresql://postgres:password@localhost:5432/fvi_analytics
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_TIMEOUT=30
```

**Important PostgreSQL Settings:**
- Enable pgvector extension for vector operations
- Set appropriate memory settings for your hardware
- Configure proper backup strategy

### Redis Configuration

```bash
# Redis settings in .env
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10
REDIS_MAX_CONNECTIONS=100
```

**Redis Memory Management:**
```bash
# Connect to Redis
docker-compose -f docker-compose.phase2.yml exec redis redis-cli

# Set memory policy
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET maxmemory 2gb

# Enable persistence
CONFIG SET save "900 1 300 10 60 10000"
```

### ML Configuration

```bash
# MLflow settings
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fvi_analytics
MLFLOW_ARTIFACT_STORE=./mlruns

# Model settings
MODEL_CACHE_SIZE=10
MODEL_CACHE_TTL=3600
DEFAULT_MODEL_VERSION=v2.1.0
```

### API Configuration

```bash
# API server settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# CORS settings
CORS_ORIGINS=*
CORS_METHODS=GET,POST,PUT,DELETE
CORS_HEADERS=*
```

### OpenTelemetry Configuration

```bash
# Tracing settings
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
JAEGER_ENDPOINT=http://localhost:14268/api/traces

# Metrics settings
PROMETHEUS_ENDPOINT=http://localhost:9090
METRICS_ENABLED=true
METRICS_PORT=8080
```

## 🧩 Phase 2 Components

### 1. Data Warehouse (dbt + PostgreSQL)

**Location:** `/dbt/`

**Models:**
- **Bronze Layer:** Raw data ingestion
  - `bronze_structured_feeds.sql` - API data
  - `bronze_gem_spider.sql` - Web crawled data
  - `bronze_document_chunks.sql` - Document processing

- **Silver Layer:** Cleaned data
  - `silver_mine_metrics.sql` - Processed metrics
  - `silver_mine_profiles.sql` - Mine information
  - `silver_document_content.sql` - Clean documents

- **Gold Layer:** Feature engineering
  - `gold_mine_features.sql` - ML-ready features

- **Mart Layer:** Business views
  - `vw_metric_dq.sql` - Data quality dashboard

**Commands:**
```bash
# Run specific model
docker-compose exec dbt-scheduler dbt run --models bronze_structured_feeds

# Test data quality
docker-compose exec dbt-scheduler dbt test

# Generate documentation
docker-compose exec dbt-scheduler dbt docs generate
docker-compose exec dbt-scheduler dbt docs serve --port 8080
```

### 2. Machine Learning Pipeline

**Components:**

**A. Training Pipeline** (`src/ml/training_pipeline.py`)
```bash
# Manual training
docker-compose exec inference-service python -m src.ml.training_pipeline

# Parameters
--experiment-name "fvi_v2"
--max-trials 100
--cv-folds 5
--test-size 0.2
```

**B. Inference Service** (`src/ml/inference_service.py`)
- ONNX runtime for <2ms predictions
- Redis caching for sub-5ms responses
- Model versioning and A/B testing

**API Endpoints:**
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mine_id": "mine_001",
    "features": {
      "production_rate": 1000,
      "equipment_health": 0.85
    }
  }'

# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"mine_id": "mine_001", "features": {...}},
      {"mine_id": "mine_002", "features": {...}}
    ]
  }'
```

**C. Dynamic Weights Engine** (`src/ml/dynamic_weights.py`)
- Nightly information gain analysis
- Spark-based feature importance calculation
- Redis persistence for real-time access

```bash
# Manual weight update
docker-compose exec weights-scheduler python -m src.ml.dynamic_weights --mode update

# View current weights
curl http://localhost:8000/weights/current

# View weight history
curl http://localhost:8000/weights/history?limit=50
```

**D. Explainability Engine** (`src/ml/explainability.py`)
- SHAP value computation
- Pre-computed explanations in Redis
- Feature importance visualization

```bash
# Get explanation
curl http://localhost:8000/explain/mine_001
```

### 3. Vector-RAG LLM System

**Location:** `src/llm/vector_rag.py`

**Features:**
- pgvector document storage
- OpenAI GPT-4o integration
- Streaming responses
- Context-aware conversations

**API Usage:**
```bash
# Chat endpoint
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest production metrics?",
    "history": []
  }'

# Streaming chat
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Analyze coal market trends"
  }'
```

### 4. Feature Store (Feast)

**Location:** `src/ml/feature_store.py`

**Setup:**
```bash
# Initialize feature store
cd feast_repo
feast init

# Apply feature definitions
feast apply

# Materialize features
feast materialize-incremental $(date -d "1 day ago" +%Y-%m-%d) $(date +%Y-%m-%d)
```

### 5. Next.js 14 Frontend

**Location:** `/frontend/`

**Development:**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

**Key Components:**
- **Dashboard:** Real-time metrics visualization
- **Chat Interface:** AI assistant with streaming
- **Weight Monitor:** Dynamic weight management
- **Mine Profiles:** Comprehensive mine data

**Available at:** http://localhost:3000

### 6. Observability Stack

**Components:**

**A. OpenTelemetry** (`src/observability/telemetry.py`)
- Distributed tracing across services
- Custom metrics for ML operations
- Performance monitoring

**B. Jaeger** (http://localhost:16686)
- Trace visualization
- Service dependency mapping
- Performance analysis

**C. Prometheus** (http://localhost:9090)
- Metrics collection
- Alerting rules
- Time-series data

**D. Grafana** (http://localhost:3001)
- Dashboard visualization
- Alert management
- Custom panels

## 🏃‍♂️ Running the System

### Complete System Startup

```bash
# 1. Start infrastructure
docker-compose -f docker-compose.phase2.yml up -d postgres redis

# 2. Wait for readiness
./scripts/wait-for-services.sh

# 3. Initialize database
docker-compose -f docker-compose.phase2.yml up -d dbt-scheduler

# 4. Start ML services
docker-compose -f docker-compose.phase2.yml up -d mlflow inference-service vector-rag-service

# 5. Start monitoring
docker-compose -f docker-compose.phase2.yml up -d prometheus grafana jaeger

# 6. Start frontend (development)
cd frontend && npm run dev

# 7. Verify all services
docker-compose -f docker-compose.phase2.yml ps
```

### Service-by-Service Startup

**Option 1: Infrastructure Only**
```bash
docker-compose -f docker-compose.phase2.yml up -d postgres redis
```

**Option 2: ML Services Only**
```bash
docker-compose -f docker-compose.phase2.yml up -d inference-service vector-rag-service
```

**Option 3: Monitoring Only**
```bash
docker-compose -f docker-compose.phase2.yml up -d prometheus grafana jaeger
```

### Development Mode

```bash
# Backend development
export ENVIRONMENT=development
python -m src.ml.inference_service

# Frontend development
cd frontend
npm run dev

# dbt development
cd dbt
dbt run --select bronze_structured_feeds
```

## 🧪 Testing and Validation

### Comprehensive Test Suite

```bash
# Run all tests
python test_phase2_implementation.py

# Test specific components
python test_phase2_implementation.py -c database
python test_phase2_implementation.py -c api
python test_phase2_implementation.py -c ml

# Generate test report
python test_phase2_implementation.py -o test_results.json
```

### Manual Testing

**Database Tests:**
```bash
# Test PostgreSQL connection
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "SELECT version();"

# Test pgvector extension
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Test dbt models
docker-compose exec dbt-scheduler dbt test
```

**API Tests:**
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:5000/health

# ML inference test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"mine_id": "test", "features": {"production_rate": 1000}}'

# Chat test
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

**Frontend Tests:**
```bash
cd frontend

# Type checking
npm run type-check

# Build test
npm run build

# Unit tests (if configured)
npm test
```

### Performance Testing

```bash
# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# ML inference performance
ab -n 100 -c 5 -p prediction_payload.json -T application/json http://localhost:8000/predict

# Database performance
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "EXPLAIN ANALYZE SELECT * FROM silver_mine_metrics LIMIT 1000;"
```

## 📚 API Documentation

### REST API Endpoints

**Base URL:** http://localhost:8000

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/explain/{mine_id}` | GET | SHAP explanations |
| `/weights/current` | GET | Current feature weights |
| `/weights/history` | GET | Weight change history |
| `/metrics` | GET | System metrics |
| `/data-quality/metrics` | GET | Data quality scores |

### Vector RAG API

**Base URL:** http://localhost:8001

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/chat` | POST | Chat completion |
| `/chat/stream` | POST | Streaming chat |
| `/documents` | GET | List documents |
| `/documents/{id}` | GET | Get document |

### MLflow API

**Base URL:** http://localhost:5000

Standard MLflow REST API for:
- Experiment management
- Model registry
- Artifact storage
- Metrics logging

### Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **MLflow UI:** http://localhost:5000

## 🔧 Troubleshooting

### Common Issues

**1. Database Connection Issues**
```bash
# Check if PostgreSQL is running
docker-compose -f docker-compose.phase2.yml ps postgres

# Check logs
docker-compose -f docker-compose.phase2.yml logs postgres

# Reset database
docker-compose -f docker-compose.phase2.yml down
docker volume rm fvi-anshu_postgres_data
docker-compose -f docker-compose.phase2.yml up -d postgres
```

**2. Redis Connection Issues**
```bash
# Check Redis status
docker-compose -f docker-compose.phase2.yml exec redis redis-cli ping

# Clear Redis cache
docker-compose -f docker-compose.phase2.yml exec redis redis-cli flushall

# Check memory usage
docker-compose -f docker-compose.phase2.yml exec redis redis-cli info memory
```

**3. ML Model Loading Issues**
```bash
# Check MLflow connectivity
curl http://localhost:5000/health

# Check model artifacts
docker-compose -f docker-compose.phase2.yml exec mlflow ls /mlruns

# Restart inference service
docker-compose -f docker-compose.phase2.yml restart inference-service
```

**4. Frontend Issues**
```bash
# Check Node.js version
node --version

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**5. Docker Issues**
```bash
# Check Docker daemon
docker info

# Clean up containers
docker-compose -f docker-compose.phase2.yml down
docker system prune -a

# Rebuild images
docker-compose -f docker-compose.phase2.yml build --no-cache
```

### Log Analysis

```bash
# View all service logs
docker-compose -f docker-compose.phase2.yml logs

# Follow specific service logs
docker-compose -f docker-compose.phase2.yml logs -f inference-service

# Search for errors
docker-compose -f docker-compose.phase2.yml logs | grep ERROR

# Export logs to file
docker-compose -f docker-compose.phase2.yml logs > system_logs.txt
```

### Performance Issues

**Database Performance:**
```bash
# Check active connections
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "SELECT count(*) FROM pg_stat_activity;"

# Check slow queries
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Analyze table sizes
docker-compose exec postgres psql -U postgres -d fvi_analytics -c "SELECT schemaname,tablename,attname,n_distinct,correlation FROM pg_stats;"
```

**Memory Issues:**
```bash
# Check container memory usage
docker stats

# Check system memory
free -h

# Check Redis memory
docker-compose exec redis redis-cli info memory
```

## 👨‍💻 Development Guide

### Development Environment Setup

```bash
# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements-phase2.txt

# Install development tools
pip install black flake8 mypy pytest pytest-cov

# Setup pre-commit hooks (optional)
pre-commit install
```

### Code Style and Standards

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Adding New Features

**1. Backend API Endpoint:**
```python
# src/api/new_endpoint.py
from fastapi import APIRouter
from src.observability.telemetry import trace_function

router = APIRouter()

@router.get("/new-feature")
@trace_function("new_feature_endpoint")
async def new_feature():
    return {"message": "New feature"}
```

**2. Frontend Component:**
```typescript
// frontend/src/components/NewFeature.tsx
'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';

export function NewFeature() {
  const { data, isLoading } = useQuery({
    queryKey: ['new-feature'],
    queryFn: () => apiClient.get('/new-feature'),
  });

  return <div>{/* Component JSX */}</div>;
}
```

**3. dbt Model:**
```sql
-- dbt/models/new_model.sql
{{ config(materialized='table') }}

SELECT 
    id,
    name,
    created_at
FROM {{ ref('bronze_source_table') }}
WHERE status = 'active'
```

### Testing New Features

```bash
# Add unit tests
# tests/test_new_feature.py

# Add integration tests
# tests/integration/test_new_feature_integration.py

# Run specific tests
pytest tests/test_new_feature.py -v

# Run with debugging
pytest tests/test_new_feature.py -v -s --pdb
```

## 🚀 Deployment

### Production Deployment

**1. Prepare Environment:**
```bash
# Create production environment file
cp .env.example .env.production

# Update with production values
nano .env.production
```

**2. Build Production Images:**
```bash
# Build all images
docker-compose -f docker-compose.phase2.yml build

# Tag for registry
docker tag fvi-analytics_inference-service:latest your-registry/fvi-inference:v2.1.0
docker tag fvi-analytics_vector-rag-service:latest your-registry/fvi-vector-rag:v2.1.0
```

**3. Deploy to Production:**
```bash
# Using Docker Compose
docker-compose -f docker-compose.phase2.yml --env-file .env.production up -d

# Using Kubernetes (if configured)
kubectl apply -f k8s/
```

### Kubernetes Deployment

```yaml
# k8s/inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fvi-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fvi-inference
  template:
    metadata:
      labels:
        app: fvi-inference
    spec:
      containers:
      - name: inference
        image: your-registry/fvi-inference:v2.1.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fvi-secrets
              key: database-url
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) handles:

1. **Code Quality:** Linting, type checking, security scans
2. **Testing:** Unit tests, integration tests, dbt tests
3. **Building:** Docker images for all services
4. **Deployment:** Automated staging and production
5. **Monitoring:** Post-deployment health checks

### Health Checks and Monitoring

```bash
# Production health check script
#!/bin/bash
services=("inference-service" "vector-rag-service" "mlflow")

for service in "${services[@]}"; do
  if curl -f http://localhost:8000/health; then
    echo "✓ $service is healthy"
  else
    echo "✗ $service is unhealthy"
    exit 1
  fi
done
```

## 📊 Monitoring

### Grafana Dashboards

**Access:** http://localhost:3001 (admin/admin)

**Key Dashboards:**
1. **System Overview:** Resource usage, service health
2. **ML Performance:** Prediction latency, model accuracy
3. **Data Quality:** Freshness, completeness, trust scores
4. **Business Metrics:** Mine performance, production trends

### Prometheus Metrics

**Access:** http://localhost:9090

**Key Metrics:**
- `fvi_requests_total` - Request count by endpoint
- `fvi_request_duration_seconds` - Request latency
- `fvi_predictions_total` - ML prediction count
- `fvi_data_quality_score` - Data quality metrics
- `fvi_errors_total` - Error count by type

### Jaeger Tracing

**Access:** http://localhost:16686

**Trace Analysis:**
- End-to-end request tracing
- Service dependency mapping
- Performance bottleneck identification
- Error propagation analysis

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
- name: fvi-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(fvi_errors_total[5m]) > 0.05
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: SlowPredictions
    expr: histogram_quantile(0.95, fvi_request_duration_seconds) > 0.1
    for: 5m
    annotations:
      summary: "Prediction latency too high"
```

### Log Management

```bash
# Centralized logging with ELK stack (optional)
docker run -d --name elasticsearch elasticsearch:7.17.0
docker run -d --name kibana kibana:7.17.0
docker run -d --name logstash logstash:7.17.0

# Configure log shipping
# /etc/filebeat/filebeat.yml
```

## 🔒 Security

### Authentication & Authorization

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in .env
SECRET_KEY=your-generated-secret-key
```

### API Security

```python
# JWT token example
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Token verification logic
    if not verify_jwt(token.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Database Security

```bash
# PostgreSQL security settings
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

### Network Security

```yaml
# Docker network isolation
networks:
  fvi-internal:
    driver: bridge
    internal: true
  fvi-external:
    driver: bridge
```

## 📖 Additional Resources

### Documentation Links

- **dbt Documentation:** Generated at http://localhost:8080 (when served)
- **API Documentation:** http://localhost:8000/docs
- **MLflow Documentation:** http://localhost:5000
- **Grafana Dashboards:** http://localhost:3001

### Learning Resources

- **dbt Learn:** https://courses.getdbt.com/
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/
- **Next.js Documentation:** https://nextjs.org/docs
- **OpenTelemetry Guide:** https://opentelemetry.io/docs/

### Community & Support

- **GitHub Issues:** Submit bugs and feature requests
- **Discord Server:** Join our community discussions
- **Email Support:** support@fvi-analytics.com

---

**Last Updated:** July 5, 2025  
**Version:** Phase 2.0.0  
**Maintained by:** FVI Analytics Team
