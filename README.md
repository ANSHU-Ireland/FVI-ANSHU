# FVI Analytics Platform

A production-grade, full-stack Future Viability Index (FVI) analytics platform for the coal industry, built with modern data engineering, machine learning, and API technologies.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd FVI-ANSHU
   pip install -r requirements.txt
   ```

2. **Start Services**
   ```bash
   # Start database
   docker-compose up -d postgres
   
   # Start API server
   python api_server.py
   ```

3. **Convert Excel Data**
   ```bash
   # Generate YAML catalog from Excel
   python scripts/excel2yaml.py "FVI Scoring Metrics_Coal.xlsx" --out meta/metric_catalogue.yaml
   ```

4. **Test System**
   ```bash
   # Run integration tests
   python test_system.py
   
   # Use CLI tool
   python simple_cli.py system status
   ```

## 📊 Current Status

### ✅ Completed Features

#### Data Management
- **Excel to YAML Conversion**: Successfully converted Excel workbook to structured YAML catalog
- **Data Sources**: 92 external data sources catalogued and structured
- **Metric Definitions**: 46 metrics across 5 thematic areas with complete metadata
- **Naming Conventions**: Standardized naming for all system components

#### Database & Infrastructure
- **Database Schema**: Complete PostgreSQL schema with all FVI entities
- **Connection Management**: Robust database connection handling with connection pooling
- **Docker Setup**: Containerized PostgreSQL and Redis services
- **Configuration Management**: Environment-based configuration with validation

#### API & Web Services
- **FastAPI Backend**: Production-ready API with automatic documentation
- **RESTful Endpoints**: Complete CRUD operations for all data entities
- **CORS Support**: Cross-origin resource sharing enabled
- **Auto Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`

#### Data Processing
- **Formula Engine**: Mathematical formulas for metric calculations
- **Dynamic Weighting**: Horizon-based weighting system (H5, H10, H20)
- **Data Validation**: Quality checks and validation rules
- **Naming Standards**: Strict naming conventions throughout

#### Command Line Interface
- **CLI Tool**: Comprehensive command-line interface for all operations
- **Database Commands**: Connection testing, initialization, status checks
- **Data Commands**: Excel conversion, validation, summary reports
- **API Commands**: Server management, endpoint testing
- **System Commands**: Integration testing, status monitoring

### 📈 Current Metrics

| Component | Count | Status |
|-----------|--------|--------|
| Data Sources | 92 | ✅ Complete |
| Metric Definitions | 46 | ✅ Complete |
| Thematic Areas | 5 | ✅ Complete |
| API Endpoints | 10+ | ✅ Working |
| Database Tables | 15+ | ✅ Schema Ready |
| CLI Commands | 20+ | ✅ Working |

### 🏗️ Architecture

```
FVI-ANSHU/
├── src/                      # Core source code
│   ├── api/                  # FastAPI application
│   ├── models/               # Database models and schemas
│   ├── database/             # Database connection and CRUD
│   ├── data/                 # Data processing and ETL
│   ├── ml/                   # Machine learning models
│   └── core/                 # Core utilities (naming, formulas)
├── scripts/                  # Automation scripts
├── meta/                     # Generated metadata
├── database/                 # Database initialization
├── airflow/                  # Airflow DAGs (future)
├── api_server.py             # Simple API server
├── simple_cli.py             # CLI tool
└── test_system.py            # Integration tests
```

## 🔧 Usage Examples

### CLI Operations
```bash
# Database operations
python simple_cli.py db test        # Test database connection
python simple_cli.py db status      # Check database status

# Data operations
python simple_cli.py data convert   # Convert Excel to YAML
python simple_cli.py data summary   # Show data summary
python simple_cli.py data validate  # Validate data quality

# API operations
python simple_cli.py api test       # Test API endpoints
python simple_cli.py api docs       # Show documentation URLs

# System operations
python simple_cli.py system test    # Run integration tests
python simple_cli.py system status  # Check all components
```

### API Usage
```bash
# Get system health
curl http://localhost:8000/health

# Get catalog summary
curl http://localhost:8000/catalog

# Get all metrics
curl http://localhost:8000/metrics

# Get specific metric
curl http://localhost:8000/metrics/01_necessity_score

# Get all data sources
curl http://localhost:8000/data-sources
```

### Python API
```python
import requests

# Get catalog summary
response = requests.get('http://localhost:8000/catalog')
catalog = response.json()
print(f"Total metrics: {catalog['total_metrics']}")

# Get specific metric
response = requests.get('http://localhost:8000/metrics/01_necessity_score')
metric = response.json()
print(f"Metric: {metric['basic_info']['title']}")
```

## 📋 Data Structure

### Metric Definition Structure
```yaml
metric_key: "01_necessity_score"
feature_column: "f_01_necessity_score"
sheet_info:
  sheet_name: "1 Necessity Score (Core)"
  sheet_number: "01"
  thematic_focus: "Social & economic indispensability of coal today"
basic_info:
  title: "NECESSITY SCORE"
  slug: "necessity_score"
  formula: "TBA Combination of metrics below"
  weighting_in_metric: 0.0
data_quality:
  structured_availability: 0
  country_level_availability: 0
  # ... other quality metrics
weight_columns:
  H5: "w_01_necessity_score_H5"
  H10: "w_01_necessity_score_H10"
  H20: "w_01_necessity_score_H20"
```

### Thematic Areas
1. **Necessity Score (Core)**: 6 metrics
2. **Emissions Score**: 7 metrics
3. **Economic Score**: 8 metrics
4. **Ecological Score**: 21 metrics
5. **Workforce Transition Score**: 4 metrics

## 🔮 Next Steps

### Immediate (Next Sprint)
- [ ] **Database Population**: Populate database with real data from YAML catalog
- [ ] **ML Pipeline**: Implement machine learning models for FVI scoring
- [ ] **Scenario Analysis**: Build scenario modeling capabilities
- [ ] **LLM Integration**: Add OpenAI-powered chat interface
- [ ] **Frontend**: Create React/Next.js dashboard

### Medium Term
- [ ] **Real-time Data**: Integrate with external APIs for live data
- [ ] **Advanced Analytics**: Implement Bayesian modeling and uncertainty quantification
- [ ] **Visualization**: Create interactive charts and dashboards
- [ ] **Export Features**: PDF reports, Excel exports
- [ ] **User Management**: Authentication and authorization

### Long Term
- [ ] **Multi-industry Support**: Extend beyond coal to other industries
- [ ] **Cloud Deployment**: Deploy to AWS/Azure/GCP
- [ ] **Monitoring**: Production monitoring and alerting
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **API Versioning**: Version management for API changes

## 🧪 Testing

### Integration Tests
```bash
# Run all tests
python test_system.py

# Expected output:
# ✅ Catalog Generation: PASS
# ✅ API Endpoints: PASS
# ✅ Data Quality: PASS
# ✅ Naming Conventions: PASS
```

### API Testing
```bash
# Test API health
curl http://localhost:8000/health

# Test API documentation
curl http://localhost:8000/docs
```

### Database Testing
```bash
# Test database connection
python simple_cli.py db test
```

## 🐛 Known Issues

1. **Import Path Issues**: Some advanced features have relative import issues
2. **ML Models**: Machine learning pipeline not yet connected to real data
3. **Chat Interface**: LLM integration requires OpenAI API key
4. **Frontend**: No UI dashboard yet (API only)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_system.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with FastAPI, PostgreSQL, Redis, and modern Python ecosystem
- Excel data processing with pandas and openpyxl
- Future ML capabilities with scikit-learn, LightGBM, and others
- Containerization with Docker and Docker Compose

---

**Status**: ✅ Phase 1 Complete - Core infrastructure and data processing working
**Next Phase**: ML pipeline and frontend development
