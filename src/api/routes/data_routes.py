from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import logging
from datetime import datetime
import yaml
import json

from ...database import get_db, DataSourceCRUD, MetricDefinitionCRUD, MetricValueCRUD
from ...models import (
    DataSourceCreate, DataSourceResponse,
    MetricDefinitionCreate, MetricDefinitionResponse,
    MetricValueCreate, MetricValueResponse
)
from ...data import ExcelProcessor, DataSourceManager, MockDataGenerator
from ...core import FVINamingConventions, FVIMetricFormulas, FormulaEngine, DynamicWeightingEngine
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload-excel", response_model=Dict[str, Any])
async def upload_excel_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process Excel file with FVI metrics."""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")
    
    try:
        # Save uploaded file temporarily
        file_path = f"{settings.DATA_DIR}/temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process Excel file
        processor = ExcelProcessor(file_path)
        results = processor.process_all_sheets()
        
        # Add processing task to background
        background_tasks.add_task(
            process_excel_data,
            results,
            db
        )
        
        return {
            "message": "Excel file uploaded and processing started",
            "filename": file.filename,
            "data_sources_count": len(results["data_sources"]),
            "metrics_count": len(results["metrics"]),
            "sheets_processed": list(results["sheet_mapping"].keys()),
            "processing_status": "started"
        }
    
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_excel_data(results: Dict[str, Any], db: Session):
    """Background task to process Excel data."""
    try:
        # Create data sources
        for data_source in results["data_sources"]:
            try:
                # Check if data source already exists
                existing = DataSourceCRUD.get_by_name(db, data_source.name)
                if not existing:
                    DataSourceCRUD.create(db, data_source)
                    logger.info(f"Created data source: {data_source.name}")
                else:
                    logger.info(f"Data source already exists: {data_source.name}")
            except Exception as e:
                logger.error(f"Error creating data source {data_source.name}: {e}")
        
        # Create metric definitions
        for metric in results["metrics"]:
            try:
                # Check if metric already exists
                existing = MetricDefinitionCRUD.get_by_slug(db, metric.metric_slug)
                if not existing:
                    MetricDefinitionCRUD.create(db, metric)
                    logger.info(f"Created metric: {metric.metric_slug}")
                else:
                    logger.info(f"Metric already exists: {metric.metric_slug}")
            except Exception as e:
                logger.error(f"Error creating metric {metric.metric_slug}: {e}")
        
        logger.info("Excel data processing completed")
    
    except Exception as e:
        logger.error(f"Error in background Excel processing: {e}")


@router.get("/fetch-external-data", response_model=Dict[str, Any])
async def fetch_external_data(
    source_name: str,
    dataset: Optional[str] = None,
    symbol: Optional[str] = None,
    country: Optional[str] = None,
    indicator: Optional[str] = None,
    background_tasks: BackgroundTasks
):
    """Fetch data from external sources."""
    try:
        data_manager = DataSourceManager()
        
        # Prepare request parameters
        request_params = {}
        if dataset:
            request_params["dataset"] = dataset
        if symbol:
            request_params["symbol"] = symbol
        if country:
            request_params["country"] = country
        if indicator:
            request_params["indicator"] = indicator
        
        # Fetch data
        result = await data_manager.fetch_from_source(source_name, **request_params)
        
        # Add background task to process and store data
        background_tasks.add_task(
            process_external_data,
            result,
            source_name
        )
        
        return {
            "message": "Data fetch initiated",
            "source": source_name,
            "parameters": request_params,
            "status": "success" if "error" not in result else "error",
            "data_preview": result.get("data", [])[:5] if "data" in result else None
        }
    
    except Exception as e:
        logger.error(f"Error fetching external data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_external_data(result: Dict[str, Any], source_name: str):
    """Background task to process external data."""
    try:
        if "error" in result:
            logger.error(f"Error in external data from {source_name}: {result['error']}")
            return
        
        # Process and store data
        # This would typically involve:
        # 1. Normalizing data format
        # 2. Mapping to metric definitions
        # 3. Storing in database
        logger.info(f"Processing external data from {source_name}")
        
        # Example processing logic would go here
        # For now, just log the data receipt
        data_count = len(result.get("data", []))
        logger.info(f"Received {data_count} data points from {source_name}")
    
    except Exception as e:
        logger.error(f"Error processing external data: {e}")


@router.get("/generate-mock-data", response_model=Dict[str, Any])
async def generate_mock_data(
    sub_industry: str = "Coal Mining",
    country: str = "Global",
    years: Optional[List[int]] = None,
    db: Session = Depends(get_db)
):
    """Generate mock data for testing."""
    try:
        # Generate mock data
        mock_data = MockDataGenerator.generate_coal_metrics_data(
            sub_industry, country, years
        )
        
        # Create metric values in database
        metric_values_created = []
        
        for metric_name, values in mock_data["metrics"].items():
            # Get or create metric definition
            metric_def = MetricDefinitionCRUD.get_by_slug(db, metric_name)
            if not metric_def:
                # Create basic metric definition
                metric_create = MetricDefinitionCreate(
                    sheet_name="Mock Data",
                    title=metric_name.replace("_", " ").title(),
                    metric_slug=metric_name,
                    details=f"Mock data for {metric_name}",
                    readiness_status="Complete"
                )
                metric_def = MetricDefinitionCRUD.create(db, metric_create)
            
            # Create metric values
            for i, value in enumerate(values):
                metric_value = MetricValueCreate(
                    metric_definition_id=metric_def.id,
                    sub_industry=sub_industry,
                    country=country,
                    year=mock_data["years"][i],
                    raw_value=value,
                    normalized_value=value,
                    confidence_score=0.8,
                    calculation_method="mock_generated"
                )
                
                db_metric_value = MetricValueCRUD.create(db, metric_value)
                metric_values_created.append(db_metric_value.id)
        
        return {
            "message": "Mock data generated successfully",
            "sub_industry": sub_industry,
            "country": country,
            "years": mock_data["years"],
            "metrics": list(mock_data["metrics"].keys()),
            "fvi_scores": mock_data["fvi_scores"],
            "metric_values_created": len(metric_values_created),
            "weights": mock_data["weights"]
        }
    
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources-info", response_model=Dict[str, Any])
async def get_data_sources_info():
    """Get information about available data sources."""
    try:
        data_manager = DataSourceManager()
        available_sources = data_manager.get_available_sources()
        
        sources_info = {}
        for source_name in available_sources:
            sources_info[source_name] = data_manager.get_source_info(source_name)
        
        return {
            "available_sources": available_sources,
            "sources_info": sources_info,
            "total_sources": len(available_sources)
        }
    
    except Exception as e:
        logger.error(f"Error getting data sources info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/normalize-data", response_model=Dict[str, Any])
async def normalize_data(
    data: Dict[str, List[float]],
    method: str = "min_max",
    handle_missing: str = "mean"
):
    """Normalize data using specified method."""
    try:
        from ...data import DataNormalizer
        
        normalized_data = {}
        for field_name, values in data.items():
            # Handle missing values
            clean_values = DataNormalizer.handle_missing_values(values, handle_missing)
            
            # Normalize
            normalized_values = DataNormalizer.normalize_to_0_100(clean_values, method)
            
            normalized_data[field_name] = normalized_values
        
        return {
            "message": "Data normalized successfully",
            "method": method,
            "missing_value_handling": handle_missing,
            "fields_processed": list(data.keys()),
            "normalized_data": normalized_data
        }
    
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export-data", response_model=Dict[str, Any])
async def export_data(
    format: str = "csv",
    sub_industry: Optional[str] = None,
    country: Optional[str] = None,
    year: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Export data in specified format."""
    try:
        # Get metric values based on filters
        metric_values = MetricValueCRUD.get_multi(db, skip=0, limit=10000)
        
        # Apply filters
        if sub_industry:
            metric_values = [mv for mv in metric_values if mv.sub_industry == sub_industry]
        if country:
            metric_values = [mv for mv in metric_values if mv.country == country]
        if year:
            metric_values = [mv for mv in metric_values if mv.year == year]
        
        # Convert to DataFrame
        data_rows = []
        for mv in metric_values:
            data_rows.append({
                "metric_id": mv.metric_definition_id,
                "sub_industry": mv.sub_industry,
                "country": mv.country,
                "year": mv.year,
                "raw_value": mv.raw_value,
                "normalized_value": mv.normalized_value,
                "confidence_score": mv.confidence_score,
                "calculation_method": mv.calculation_method
            })
        
        df = pd.DataFrame(data_rows)
        
        # Export based on format
        if format == "csv":
            output = io.StringIO()
            df.to_csv(output, index=False)
            content = output.getvalue()
        elif format == "json":
            content = df.to_json(orient="records")
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        return {
            "message": "Data exported successfully",
            "format": format,
            "records_count": len(data_rows),
            "filters_applied": {
                "sub_industry": sub_industry,
                "country": country,
                "year": year
            },
            "data": content
        }
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert-excel-to-yaml", response_model=Dict[str, Any])
async def convert_excel_to_yaml(
    file: UploadFile = File(...),
    output_path: Optional[str] = None
):
    """Convert Excel file to YAML metric catalog using FVI naming conventions."""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"{settings.DATA_DIR}/temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Convert to YAML using the script logic
        from ...scripts.excel2yaml import ExcelToYAMLConverter
        
        converter = ExcelToYAMLConverter(temp_path)
        catalog = converter.convert_to_yaml()
        
        # Save YAML if output path provided
        if output_path:
            converter.save_yaml(output_path)
        
        return {
            "message": "Excel converted to YAML successfully",
            "filename": file.filename,
            "catalog_summary": catalog["summary"],
            "naming_conventions_applied": True,
            "yaml_catalog": catalog if not output_path else None,
            "output_path": output_path
        }
    
    except Exception as e:
        logger.error(f"Error converting Excel to YAML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-metrics", response_model=Dict[str, Any])
async def calculate_metrics(
    data: Dict[str, Any],
    sheet_number: Optional[str] = None,
    use_formulas: bool = True
):
    """Calculate FVI metrics using the formula engine."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data.get("data", []))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Initialize formula engine
        formula_engine = FormulaEngine()
        
        # Calculate metrics
        if sheet_number:
            # Calculate metrics for specific sheet
            results_df = formula_engine.calculate_sheet_metrics(sheet_number, df)
        else:
            # Calculate all metrics
            results_df = formula_engine.calculate_all_metrics(df)
        
        # Extract feature columns
        feature_cols = [col for col in results_df.columns if col.startswith('f_')]
        
        return {
            "message": "Metrics calculated successfully",
            "input_rows": len(df),
            "feature_columns_generated": len(feature_cols),
            "feature_columns": feature_cols,
            "results": results_df.to_dict('records'),
            "sheet_number": sheet_number,
            "formulas_used": use_formulas
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-dynamic-weights", response_model=Dict[str, Any])
async def calculate_dynamic_weights(
    data: Dict[str, Any],
    horizon: str = "H10",
    include_user_overrides: bool = False
):
    """Calculate dynamic weights based on data quality and information gain."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data.get("data", []))
        target_col = data.get("target_column", "fvi_score")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
        
        # Separate features and target
        target = df[target_col]
        features = df.drop(columns=[target_col])
        
        # Initialize weighting engine
        weighting_engine = DynamicWeightingEngine()
        
        # Set base weights if provided
        base_weights = data.get("base_weights", {})
        if base_weights:
            weighting_engine.set_base_weights(base_weights)
        
        # Calculate dynamic weights
        dynamic_weights = weighting_engine.calculate_dynamic_weights(features, target, horizon)
        
        # Apply user overrides if provided
        if include_user_overrides and "user_overrides" in data:
            dynamic_weights = weighting_engine.update_weights_with_user_overrides(
                dynamic_weights, data["user_overrides"]
            )
        
        # Normalize weights
        normalized_weights = weighting_engine.normalize_weights(dynamic_weights)
        
        # Get component weights for transparency
        quality_weights = weighting_engine.quality_weights
        info_gain_weights = weighting_engine.info_gain_weights
        
        return {
            "message": "Dynamic weights calculated successfully",
            "horizon": horizon,
            "dynamic_weights": dynamic_weights,
            "normalized_weights": normalized_weights,
            "component_weights": {
                "quality_weights": quality_weights,
                "info_gain_weights": info_gain_weights
            },
            "base_weights": base_weights,
            "user_overrides_applied": include_user_overrides
        }
    
    except Exception as e:
        logger.error(f"Error calculating dynamic weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate-naming-conventions", response_model=Dict[str, Any])
async def validate_naming_conventions(
    metric_key: Optional[str] = None,
    dataset_table: Optional[str] = None,
    feature_column: Optional[str] = None,
    weight_column: Optional[str] = None,
    composite_score: Optional[str] = None
):
    """Validate FVI naming conventions."""
    try:
        validations = {}
        
        if metric_key:
            validations["metric_key"] = {
                "value": metric_key,
                "valid": FVINamingConventions.validate_metric_key(metric_key),
                "parsed": FVINamingConventions.parse_metric_key(metric_key)
            }
        
        if dataset_table:
            validations["dataset_table"] = {
                "value": dataset_table,
                "valid": FVINamingConventions.validate_dataset_table(dataset_table)
            }
        
        if feature_column:
            validations["feature_column"] = {
                "value": feature_column,
                "valid": FVINamingConventions.validate_feature_column(feature_column)
            }
        
        if weight_column:
            validations["weight_column"] = {
                "value": weight_column,
                "valid": FVINamingConventions.validate_weight_column(weight_column),
                "parsed": FVINamingConventions.parse_weight_column(weight_column)
            }
        
        if composite_score:
            validations["composite_score"] = {
                "value": composite_score,
                "valid": FVINamingConventions.validate_composite_score(composite_score),
                "parsed": FVINamingConventions.parse_composite_score(composite_score)
            }
        
        return {
            "message": "Naming convention validation completed",
            "validations": validations,
            "naming_patterns": {
                "metric_key": FVINamingConventions.METRIC_KEY_PATTERN,
                "dataset_table": FVINamingConventions.DATASET_TABLE_PATTERN,
                "feature_column": FVINamingConventions.FEATURE_COLUMN_PATTERN,
                "weight_column": FVINamingConventions.WEIGHT_COLUMN_PATTERN,
                "composite_score": FVINamingConventions.COMPOSITE_SCORE_PATTERN
            }
        }
    
    except Exception as e:
        logger.error(f"Error validating naming conventions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate-standardized-names", response_model=Dict[str, Any])
async def generate_standardized_names(
    sheet_name: str,
    slug: str,
    subindustry: str = "coal",
    source: Optional[str] = None,
    topic: Optional[str] = None
):
    """Generate standardized names following FVI conventions."""
    try:
        # Generate metric key
        metric_key = FVINamingConventions.generate_metric_key(sheet_name, slug)
        
        # Generate feature column
        feature_column = FVINamingConventions.generate_feature_column(metric_key)
        
        # Generate weight columns for all horizons
        weight_columns = {}
        for horizon in FVIHorizon:
            weight_columns[horizon.value] = FVINamingConventions.generate_weight_column(metric_key, horizon)
        
        # Generate composite scores for all horizons
        composite_scores = {}
        for horizon in FVIHorizon:
            composite_scores[horizon.value] = FVINamingConventions.generate_composite_score(subindustry, horizon)
        
        # Generate dataset table if source and topic provided
        dataset_table = None
        if source and topic:
            dataset_table = FVINamingConventions.generate_dataset_table(source, topic)
        
        return {
            "message": "Standardized names generated successfully",
            "input": {
                "sheet_name": sheet_name,
                "slug": slug,
                "subindustry": subindustry,
                "source": source,
                "topic": topic
            },
            "generated_names": {
                "metric_key": metric_key,
                "feature_column": feature_column,
                "weight_columns": weight_columns,
                "composite_scores": composite_scores,
                "dataset_table": dataset_table
            }
        }
    
    except Exception as e:
        logger.error(f"Error generating standardized names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formula-catalog", response_model=Dict[str, Any])
async def get_formula_catalog(
    sheet_number: Optional[str] = None
):
    """Get FVI formula catalog."""
    try:
        from ...core.naming import FVIMetricFormulas
        
        if sheet_number:
            # Get formulas for specific sheet
            formulas = FVIMetricFormulas.get_sheet_formulas(sheet_number)
            return {
                "message": f"Formulas retrieved for sheet {sheet_number}",
                "sheet_number": sheet_number,
                "formulas": formulas,
                "count": len(formulas)
            }
        else:
            # Get all formulas
            all_formulas = FVIMetricFormulas.list_all_formulas()
            return {
                "message": "All formulas retrieved",
                "formulas_by_sheet": all_formulas,
                "total_sheets": len(all_formulas),
                "total_formulas": sum(len(formulas) for formulas in all_formulas.values())
            }
    
    except Exception as e:
        logger.error(f"Error retrieving formula catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-formula", response_model=Dict[str, Any])
async def validate_formula(
    formula: str,
    sample_data: Dict[str, Any]
):
    """Validate a formula with sample data."""
    try:
        formula_engine = FormulaEngine()
        
        # Validate the formula
        is_valid = formula_engine.validate_formula(formula, sample_data)
        
        # Get required variables
        variables = formula_engine.get_formula_variables(formula)
        
        # Check if all required variables are present
        missing_variables = [var for var in variables if var not in sample_data]
        
        # Try to evaluate if valid
        result = None
        error = None
        if is_valid:
            try:
                result = formula_engine.evaluate_formula(formula, sample_data)
            except Exception as e:
                error = str(e)
        
        return {
            "message": "Formula validation completed",
            "formula": formula,
            "is_valid": is_valid,
            "required_variables": variables,
            "missing_variables": missing_variables,
            "sample_data": sample_data,
            "evaluation_result": result,
            "evaluation_error": error
        }
    
    except Exception as e:
        logger.error(f"Error validating formula: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/excel-to-yaml", response_model=Dict[str, Any])
async def convert_excel_to_yaml(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Convert Excel workbook to structured YAML format following FVI naming conventions."""
    try:
        # Read Excel file
        content = await file.read()
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
        
        # Initialize processors
        naming_conv = FVINamingConventions()
        processor = ExcelProcessor(naming_conv)
        
        # Process Excel data
        yaml_data = processor.excel_to_yaml(excel_data)
        
        # Validate naming conventions
        validation_results = naming_conv.validate_data_structure(yaml_data)
        
        return {
            "success": True,
            "yaml_data": yaml_data,
            "validation": validation_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Excel to YAML conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")


@router.post("/calculate-formulas", response_model=Dict[str, Any])
async def calculate_formulas(
    data: Dict[str, Any],
    formula_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Calculate metrics using formula-driven feature engineering."""
    try:
        # Initialize formula engine
        formula_engine = FormulaEngine()
        
        # Calculate formulas
        if formula_name:
            # Calculate specific formula
            result = formula_engine.calculate_single_formula(data, formula_name)
        else:
            # Calculate all applicable formulas
            result = formula_engine.calculate_all_formulas(data)
        
        return {
            "success": True,
            "calculations": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Formula calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Formula calculation failed: {str(e)}")


@router.post("/dynamic-weights", response_model=Dict[str, Any])
async def calculate_dynamic_weights(
    scenario_data: Dict[str, Any],
    weight_config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Calculate dynamic weights based on scenario analysis."""
    try:
        # Initialize dynamic weighting engine
        weighting_engine = DynamicWeightingEngine()
        
        # Calculate dynamic weights
        weights = weighting_engine.calculate_scenario_weights(scenario_data, weight_config)
        
        # Get weight explanations
        explanations = weighting_engine.get_weight_explanations(weights)
        
        return {
            "success": True,
            "weights": weights,
            "explanations": explanations,
            "scenario_summary": weighting_engine.get_scenario_summary(scenario_data),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Dynamic weight calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dynamic weight calculation failed: {str(e)}")


@router.post("/validate-naming", response_model=Dict[str, Any])
async def validate_naming_conventions(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Validate data structure against FVI naming conventions."""
    try:
        naming_conv = FVINamingConventions()
        validation_results = naming_conv.validate_data_structure(data)
        
        return {
            "success": True,
            "validation": validation_results,
            "is_valid": validation_results.get("is_valid", False),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Naming validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Naming validation failed: {str(e)}")


@router.get("/formula-catalog", response_model=Dict[str, Any])
async def get_formula_catalog(
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get the complete formula catalog with documentation."""
    try:
        formulas = FVIMetricFormulas()
        
        if category:
            catalog = formulas.get_formulas_by_category(category)
        else:
            catalog = formulas.get_all_formulas()
        
        return {
            "success": True,
            "catalog": catalog,
            "categories": formulas.get_categories(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Formula catalog retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Formula catalog retrieval failed: {str(e)}")


@router.post("/process-excel-formulas", response_model=Dict[str, Any])
async def process_excel_with_formulas(
    file: UploadFile = File(...),
    apply_formulas: bool = True,
    calculate_weights: bool = True,
    db: Session = Depends(get_db)
):
    """Process Excel file with formula-driven feature engineering and dynamic weighting."""
    try:
        # Read Excel file
        content = await file.read()
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
        
        # Initialize components
        naming_conv = FVINamingConventions()
        processor = ExcelProcessor(naming_conv)
        formula_engine = FormulaEngine()
        weighting_engine = DynamicWeightingEngine()
        
        # Process Excel data
        processed_data = processor.process_workbook(excel_data)
        
        # Apply formulas if requested
        if apply_formulas:
            formula_results = formula_engine.calculate_all_formulas(processed_data)
            processed_data.update(formula_results)
        
        # Calculate dynamic weights if requested
        weights = None
        if calculate_weights:
            weights = weighting_engine.calculate_scenario_weights(processed_data)
        
        # Validate naming conventions
        validation_results = naming_conv.validate_data_structure(processed_data)
        
        return {
            "success": True,
            "processed_data": processed_data,
            "formula_results": formula_results if apply_formulas else None,
            "dynamic_weights": weights if calculate_weights else None,
            "validation": validation_results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Excel processing with formulas failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excel processing failed: {str(e)}")


@router.post("/update-weights", response_model=Dict[str, Any])
async def update_metric_weights(
    weight_updates: Dict[str, float],
    scenario_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update metric weights with validation and historical tracking."""
    try:
        weighting_engine = DynamicWeightingEngine()
        
        # Validate weight updates
        validation_results = weighting_engine.validate_weight_updates(weight_updates)
        
        if not validation_results.get("is_valid", False):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid weight updates: {validation_results.get('errors', [])}"
            )
        
        # Apply weight updates
        updated_weights = weighting_engine.apply_weight_updates(weight_updates, scenario_id)
        
        # Store in database (would implement CRUD operations)
        # weight_crud = WeightCRUD(db)
        # weight_crud.create_weight_update(updated_weights, scenario_id)
        
        return {
            "success": True,
            "updated_weights": updated_weights,
            "validation": validation_results,
            "scenario_id": scenario_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Weight update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weight update failed: {str(e)}")


@router.get("/weight-history/{scenario_id}", response_model=Dict[str, Any])
async def get_weight_history(
    scenario_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get historical weight changes for a scenario."""
    try:
        # Would implement with actual database queries
        # weight_crud = WeightCRUD(db)
        # history = weight_crud.get_weight_history(scenario_id, limit)
        
        # Mock response for now
        history = {
            "scenario_id": scenario_id,
            "changes": [],
            "current_weights": {},
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Weight history retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weight history retrieval failed: {str(e)}")


@router.post("/optimize-weights", response_model=Dict[str, Any])
async def optimize_weights(
    optimization_config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Optimize weights using machine learning techniques."""
    try:
        weighting_engine = DynamicWeightingEngine()
        
        # Perform weight optimization
        optimized_weights = weighting_engine.optimize_weights(optimization_config)
        
        # Get optimization metrics
        optimization_metrics = weighting_engine.get_optimization_metrics(optimized_weights)
        
        return {
            "success": True,
            "optimized_weights": optimized_weights,
            "optimization_metrics": optimization_metrics,
            "config": optimization_config,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Weight optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Weight optimization failed: {str(e)}")


@router.post("/batch-formulas", response_model=Dict[str, Any])
async def batch_calculate_formulas(
    batch_data: List[Dict[str, Any]],
    formula_names: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """Calculate formulas for multiple data records in batch."""
    try:
        formula_engine = FormulaEngine()
        
        # Process batch data
        batch_results = []
        for i, data in enumerate(batch_data):
            try:
                if formula_names:
                    result = formula_engine.calculate_specific_formulas(data, formula_names)
                else:
                    result = formula_engine.calculate_all_formulas(data)
                
                batch_results.append({
                    "index": i,
                    "success": True,
                    "results": result
                })
            except Exception as e:
                batch_results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "batch_results": batch_results,
            "total_processed": len(batch_data),
            "successful": len([r for r in batch_results if r["success"]]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch formula calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch calculation failed: {str(e)}")


@router.get("/naming-conventions", response_model=Dict[str, Any])
async def get_naming_conventions(
    db: Session = Depends(get_db)
):
    """Get the complete FVI naming conventions and standards."""
    try:
        naming_conv = FVINamingConventions()
        conventions = naming_conv.get_all_conventions()
        
        return {
            "success": True,
            "conventions": conventions,
            "version": naming_conv.get_version(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Naming conventions retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Naming conventions retrieval failed: {str(e)}")


@router.post("/validate-metric-definitions", response_model=Dict[str, Any])
async def validate_metric_definitions(
    metric_definitions: List[Dict[str, Any]],
    db: Session = Depends(get_db)
):
    """Validate metric definitions against naming conventions and formula catalog."""
    try:
        naming_conv = FVINamingConventions()
        formulas = FVIMetricFormulas()
        
        validation_results = []
        for metric_def in metric_definitions:
            # Validate naming
            naming_result = naming_conv.validate_metric_definition(metric_def)
            
            # Validate formula if present
            formula_result = None
            if metric_def.get("formula"):
                formula_result = formulas.validate_formula(metric_def["formula"])
            
            validation_results.append({
                "metric": metric_def.get("name", "unknown"),
                "naming_validation": naming_result,
                "formula_validation": formula_result,
                "is_valid": naming_result.get("is_valid", False) and 
                          (formula_result is None or formula_result.get("is_valid", False))
            })
        
        return {
            "success": True,
            "validation_results": validation_results,
            "total_validated": len(metric_definitions),
            "valid_count": len([r for r in validation_results if r["is_valid"]]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Metric definition validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
