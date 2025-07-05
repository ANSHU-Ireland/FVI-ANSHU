# Phase 2 Implementation Guide

This document provides a comprehensive guide for implementing Phase 2 of the FVI Analytics Platform.

## Architecture Overview

Phase 2 transforms the prototype into a production-grade, low-latency, explainable system with the following components:

### 1. Data Warehouse (dbt + PostgreSQL)
- **Bronze Layer**: Raw data ingestion from structured feeds, GEM spider, and document chunks
- **Silver Layer**: Cleaned and typed data with quality scoring
- **Gold Layer**: Feature engineering and aggregation
- **Mart Layer**: Business-ready views and metrics

### 2. Machine Learning Pipeline
- **Dynamic Weight Engine**: Nightly information gain analysis with Redis persistence
- **Training Pipeline**: MLflow + Optuna + LightGBM with model versioning
- **Inference Service**: ONNX runtime with <2ms predictions and Redis caching
- **Explainability**: SHAP values with pre-computation and caching

### 3. Vector-RAG LLM Layer
- **pgvector**: Vector storage for document embeddings
- **OpenAI GPT-4o**: Chat interface with streaming responses
- **Retrieval**: Semantic search with relevance scoring
- **Guardrails**: Content filtering and safety checks

### 4. Feature Store (Feast)
- **Offline Store**: Historical features for training
- **Online Store**: Real-time features for inference
- **Feature Definitions**: Centralized schema management
- **Monitoring**: Feature drift detection

### 5. Frontend (Next.js 14)
- **Dashboard**: Real-time metrics and visualizations
- **Chat Interface**: Natural language queries with streaming
- **Weight Monitor**: Dynamic weight visualization
- **Observability**: Distributed tracing integration

### 6. Observability Stack
- **OpenTelemetry**: Distributed tracing and metrics
- **Jaeger**: Trace visualization
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and alerting

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.10+ (for local development)
- 16GB RAM recommended
- 50GB+ free disk space

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd fvi-analytics

# Copy environment configuration
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Start Infrastructure

```bash
# Start all services
docker-compose -f docker-compose.phase2.yml up -d

# Check service health
docker-compose -f docker-compose.phase2.yml ps
```

### 3. Initialize Database

```bash
# Run database migrations
docker-compose -f docker-compose.phase2.yml exec postgres psql -U postgres -d fvi_analytics -f /docker-entrypoint-initdb.d/01_create_tables.sql

# Run dbt models
docker-compose -f docker-compose.phase2.yml exec dbt-scheduler dbt run --project-dir /dbt
```

### 4. Start Frontend (Development)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 5. Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | Main application |
| API Server | http://localhost:8000 | REST API |
| Vector RAG | http://localhost:8001 | Chat API |
| MLflow | http://localhost:5000 | Model tracking |
| Grafana | http://localhost:3001 | Monitoring |
| Jaeger | http://localhost:16686 | Distributed tracing |

## Development Workflow

### 1. Data Pipeline Development

```bash
# Test dbt models
cd dbt
dbt test --project-dir .

# Run specific model
dbt run --project-dir . --models silver_mine_metrics

# Generate documentation
dbt docs generate --project-dir .
dbt docs serve --project-dir .
```

### 2. ML Pipeline Development

```bash
# Train new model
python -m src.ml.training_pipeline --experiment-name "fvi_v2"

# Update dynamic weights
python -m src.ml.dynamic_weights --mode update

# Test inference
python -m src.ml.inference_service --mine-id "mine_001" --features '{"production_rate": 1000}'
```

### 3. Frontend Development

```bash
cd frontend

# Run development server
npm run dev

# Type checking
npm run type-check

# Build for production
npm run build
```

### 4. Testing

```bash
# Run Python tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Test API endpoints
curl http://localhost:8000/health

# Test chat interface
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest production metrics?"}'
```

## Configuration

### Database Configuration
- **Connection Pool**: 20 connections with 30 overflow
- **Timeout**: 30 seconds for queries
- **SSL Mode**: Required in production

### Redis Configuration
- **Memory Policy**: allkeys-lru
- **Max Memory**: 2GB
- **Persistence**: AOF enabled

### ML Configuration
- **Model Cache**: 10 models in memory
- **Inference Timeout**: 5 seconds
- **Batch Size**: 100 predictions max

### Monitoring Configuration
- **Metrics Retention**: 30 days
- **Trace Sampling**: 10% in production
- **Log Level**: INFO in production

## Deployment

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.phase2.yml build

# Deploy with secrets
docker-compose -f docker-compose.phase2.yml --env-file .env.production up -d

# Verify deployment
docker-compose -f docker-compose.phase2.yml exec inference-service curl http://localhost:8000/health
```

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:

1. **Code Quality**: Linting, type checking, security scanning
2. **Testing**: Unit tests, integration tests, dbt tests
3. **Build**: Docker images for all services
4. **Deploy**: Staging and production deployment
5. **Monitoring**: Performance tests and health checks

### Scaling Considerations

- **Horizontal Scaling**: Add more inference service replicas
- **Database Sharding**: Consider partitioning by mine_id
- **Redis Clustering**: For high availability
- **CDN**: For static assets and model artifacts

## Monitoring & Observability

### Key Metrics
- **Prediction Latency**: <2ms p95
- **Data Quality Score**: >80%
- **Model Accuracy**: >85%
- **System Availability**: >99.9%

### Alerting Rules
- High error rate (>5%)
- Low data quality score (<70%)
- Model drift detected
- Infrastructure resource limits

### Dashboards
- **Operations**: System health, performance metrics
- **Business**: Model predictions, data quality
- **Development**: API usage, feature performance

## Security

### Authentication
- JWT tokens for API access
- Service-to-service authentication
- Rate limiting per endpoint

### Data Protection
- Encryption at rest and in transit
- PII data anonymization
- Audit logging for compliance

### Network Security
- VPC isolation
- Firewall rules
- SSL/TLS termination

## Troubleshooting

### Common Issues

1. **Database Connection Timeout**
   ```bash
   # Check database logs
   docker-compose -f docker-compose.phase2.yml logs postgres
   
   # Reset connections
   docker-compose -f docker-compose.phase2.yml restart postgres
   ```

2. **Redis Memory Issues**
   ```bash
   # Check Redis memory usage
   docker-compose -f docker-compose.phase2.yml exec redis redis-cli info memory
   
   # Clear cache
   docker-compose -f docker-compose.phase2.yml exec redis redis-cli flushall
   ```

3. **Model Loading Errors**
   ```bash
   # Check MLflow connectivity
   docker-compose -f docker-compose.phase2.yml exec inference-service curl http://mlflow:5000/health
   
   # Reload models
   docker-compose -f docker-compose.phase2.yml restart inference-service
   ```

### Log Analysis

```bash
# View service logs
docker-compose -f docker-compose.phase2.yml logs -f inference-service

# Search for errors
docker-compose -f docker-compose.phase2.yml logs | grep ERROR

# Check specific time range
docker-compose -f docker-compose.phase2.yml logs --since 2023-01-01T00:00:00Z
```

## Performance Optimization

### Database Optimization
- Index optimization for frequent queries
- Query plan analysis
- Connection pooling tuning

### Caching Strategy
- Redis for hot data
- Application-level caching
- CDN for static assets

### ML Optimization
- Model quantization
- Batch inference
- Feature caching

## Maintenance

### Regular Tasks
- **Daily**: Health checks, backup verification
- **Weekly**: Model retraining, data quality reports
- **Monthly**: Performance reviews, capacity planning

### Backup Strategy
- **Database**: Daily incremental, weekly full
- **Models**: Versioned in MLflow
- **Configuration**: Git-based versioning

### Update Process
1. Deploy to staging environment
2. Run integration tests
3. Gradual rollout to production
4. Monitor key metrics
5. Rollback if issues detected

## Support

For issues and questions:
- Check logs and monitoring dashboards
- Review this documentation
- Contact the FVI Analytics team
- Submit issues via GitHub

## Next Steps

Phase 3 roadmap includes:
- Advanced analytics and insights
- Multi-model ensembles
- Real-time streaming pipeline
- Enhanced security features
- Mobile application

---

*This document is maintained by the FVI Analytics Team. Last updated: 2025-07-05*
