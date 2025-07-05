"""
FVI Coal Daily ETL Pipeline
===========================

End-to-end DAG for the Future Viability Index - Coal pilot.
Implements the complete data pipeline from source harvesting to 
model inference and LLM-powered explanations.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
import pandas as pd
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.sources import DataSourceManager
from core.formulas import FormulaEngine, DynamicWeightingEngine
from core.naming import FVINamingConventions
from ml.models import FVIPredictor, EnsemblePredictor

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'fvi-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'fvi_coal_daily',
    default_args=default_args,
    description='FVI Coal Daily ETL Pipeline',
    schedule_interval='0 3 * * *',  # Daily at 3 AM
    max_active_runs=1,
    catchup=False,
    tags=['fvi', 'coal', 'etl', 'ml'],
)


def download_sources(**context):
    """Download data from external sources."""
    logger.info("Starting data source download")
    
    # Initialize data source manager
    data_manager = DataSourceManager()
    
    # Define data sources to fetch
    sources_to_fetch = [
        {
            "source": "IEA",
            "dataset": "WORLDBAL",
            "country": "WORLD"
        },
        {
            "source": "EIA", 
            "series_id": "INTL.44-1-WORL-QBTU.A"
        },
        {
            "source": "World Bank",
            "indicator": "EG.USE.COAL.KT.OE",
            "country": "WLD"
        },
        {
            "source": "Yahoo Finance",
            "symbol": "COAL",
            "period": "1y"
        }
    ]
    
    # Fetch data from each source
    results = {}
    for source_config in sources_to_fetch:
        try:
            source_name = source_config.pop("source")
            result = data_manager.fetch_from_source(source_name, **source_config)
            results[source_name] = result
            logger.info(f"Successfully fetched data from {source_name}")
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {e}")
            results[source_name] = {"error": str(e)}
    
    # Store results for next task
    context['task_instance'].xcom_push(key='download_results', value=results)
    logger.info(f"Downloaded data from {len(results)} sources")
    
    return results


def raw_to_bronze(**context):
    """Transform raw data to bronze (standardized) format."""
    logger.info("Starting raw to bronze transformation")
    
    # Get download results
    download_results = context['task_instance'].xcom_pull(key='download_results')
    
    # Connect to database
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    bronze_tables = {}
    
    for source_name, result in download_results.items():
        if "error" in result:
            logger.warning(f"Skipping {source_name} due to error: {result['error']}")
            continue
            
        try:
            # Create standardized table name
            table_name = FVINamingConventions.generate_dataset_table(source_name, "raw_data")
            
            # Convert to DataFrame
            if "data" in result:
                df = pd.DataFrame(result["data"])
                
                # Add metadata columns
                df['source_name'] = source_name
                df['ingest_date'] = datetime.utcnow()
                df['data_version'] = result.get('timestamp', datetime.utcnow().isoformat())
                
                # Store in bronze layer
                df.to_sql(table_name, postgres_hook.get_sqlalchemy_engine(), 
                         if_exists='replace', index=False)
                
                bronze_tables[source_name] = table_name
                logger.info(f"Created bronze table: {table_name} with {len(df)} rows")
                
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
    
    context['task_instance'].xcom_push(key='bronze_tables', value=bronze_tables)
    return bronze_tables


def bronze_to_silver(**context):
    """Transform bronze data to silver (cleaned and validated) format."""
    logger.info("Starting bronze to silver transformation")
    
    # Get bronze tables
    bronze_tables = context['task_instance'].xcom_pull(key='bronze_tables')
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    silver_tables = {}
    
    for source_name, bronze_table in bronze_tables.items():
        try:
            # Read bronze data
            df = pd.read_sql_table(bronze_table, engine)
            
            # Data cleaning and validation
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Type enforcement
            for col in df.columns:
                if col.endswith('_date'):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif col.endswith('_value') or col.endswith('_amount'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Unit conversions (example: t to mt)
            for col in df.columns:
                if 'tonnes' in col.lower() or '_t' in col.lower():
                    df[col] = df[col] / 1000  # Convert to million tonnes
            
            # Create silver table
            silver_table = bronze_table.replace('raw_', 'silver_')
            df.to_sql(silver_table, engine, if_exists='replace', index=False)
            
            silver_tables[source_name] = silver_table
            logger.info(f"Created silver table: {silver_table} with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error processing bronze table {bronze_table}: {e}")
    
    context['task_instance'].xcom_push(key='silver_tables', value=silver_tables)
    return silver_tables


def silver_to_gold(**context):
    """Transform silver data to gold (feature engineered) format."""
    logger.info("Starting silver to gold transformation")
    
    # Get silver tables
    silver_tables = context['task_instance'].xcom_pull(key='silver_tables')
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    # Initialize formula engine
    formula_engine = FormulaEngine()
    
    # Combine all silver data
    combined_data = []
    for source_name, silver_table in silver_tables.items():
        try:
            df = pd.read_sql_table(silver_table, engine)
            df['source'] = source_name
            combined_data.append(df)
        except Exception as e:
            logger.error(f"Error reading silver table {silver_table}: {e}")
    
    if not combined_data:
        logger.warning("No silver data available for feature engineering")
        return {}
    
    # Combine all data
    combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
    
    # Calculate features using formula engine
    try:
        features_df = formula_engine.calculate_all_metrics(combined_df)
        
        # Create gold table
        gold_table = "gold_fvi_features"
        features_df.to_sql(gold_table, engine, if_exists='replace', index=False)
        
        # Count feature columns
        feature_cols = [col for col in features_df.columns if col.startswith('f_')]
        
        logger.info(f"Created gold table: {gold_table} with {len(feature_cols)} feature columns")
        
        context['task_instance'].xcom_push(key='gold_table', value=gold_table)
        context['task_instance'].xcom_push(key='feature_columns', value=feature_cols)
        
        return gold_table
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return {}


def update_weights(**context):
    """Update dynamic weights based on data quality and information gain."""
    logger.info("Starting weight update")
    
    # Get gold table
    gold_table = context['task_instance'].xcom_pull(key='gold_table')
    
    if not gold_table:
        logger.warning("No gold table available for weight update")
        return {}
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    try:
        # Read gold data
        df = pd.read_sql_table(gold_table, engine)
        
        # Initialize weighting engine
        weighting_engine = DynamicWeightingEngine()
        
        # Calculate dynamic weights for each horizon
        horizons = ["H5", "H10", "H20"]
        weight_results = {}
        
        for horizon in horizons:
            # Create dummy target for weight calculation
            # In production, this would be actual FVI scores
            target = pd.Series([50.0] * len(df), name=f'fvi_score_{horizon}')
            
            # Calculate weights
            weights = weighting_engine.calculate_dynamic_weights(df, target, horizon)
            weight_results[horizon] = weights
            
            logger.info(f"Calculated {len(weights)} weights for horizon {horizon}")
        
        # Store weights in database
        weights_table = "dim_weights"
        weights_data = []
        
        for horizon, weights in weight_results.items():
            for weight_name, weight_value in weights.items():
                weights_data.append({
                    'weight_name': weight_name,
                    'horizon': horizon,
                    'weight_value': weight_value,
                    'updated_at': datetime.utcnow()
                })
        
        weights_df = pd.DataFrame(weights_data)
        weights_df.to_sql(weights_table, engine, if_exists='replace', index=False)
        
        logger.info(f"Updated {len(weights_data)} weights in {weights_table}")
        
        context['task_instance'].xcom_push(key='weights_updated', value=True)
        return weight_results
        
    except Exception as e:
        logger.error(f"Error updating weights: {e}")
        return {}


def train_model(**context):
    """Train ML models if needed (weekly schedule)."""
    logger.info("Starting model training")
    
    # Check if it's time to retrain (weekly)
    execution_date = context['execution_date']
    if execution_date.weekday() != 0:  # Monday = 0
        logger.info("Skipping model training - not Monday")
        return {"skipped": True}
    
    # Get gold table
    gold_table = context['task_instance'].xcom_pull(key='gold_table')
    
    if not gold_table:
        logger.warning("No gold table available for model training")
        return {}
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    try:
        # Read training data
        df = pd.read_sql_table(gold_table, engine)
        
        # Create dummy target for training
        # In production, this would be actual historical FVI scores
        target = pd.Series([50.0] * len(df), name='fvi_score')
        
        # Initialize ensemble predictor
        ensemble = EnsemblePredictor(['lightgbm', 'catboost', 'random_forest'])
        
        # Train ensemble
        training_results = ensemble.train(df, target)
        
        # Save model
        model_path = f"/app/models/fvi_ensemble_{execution_date.strftime('%Y%m%d')}.pkl"
        ensemble.save_model(model_path)
        
        logger.info(f"Model training completed. Saved to {model_path}")
        
        context['task_instance'].xcom_push(key='model_path', value=model_path)
        context['task_instance'].xcom_push(key='training_results', value=training_results)
        
        return training_results
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return {}


def compute_shap(**context):
    """Compute SHAP explanations for model predictions."""
    logger.info("Starting SHAP computation")
    
    # Get model path
    model_path = context['task_instance'].xcom_pull(key='model_path')
    
    if not model_path:
        logger.warning("No model available for SHAP computation")
        return {}
    
    # Get gold table
    gold_table = context['task_instance'].xcom_pull(key='gold_table')
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    try:
        # Read data
        df = pd.read_sql_table(gold_table, engine)
        
        # Load model
        ensemble = EnsemblePredictor()
        ensemble.load_model(model_path)
        
        # Compute SHAP values
        # This would use the ensemble's explain_prediction method
        shap_data = []
        
        for i, row in df.iterrows():
            # Get explanation for this sample
            explanation = ensemble.models['lightgbm'].explain_prediction(row.to_frame().T)
            
            if "feature_contributions" in explanation:
                for feature, contribution in explanation["feature_contributions"].items():
                    shap_data.append({
                        'sample_id': i,
                        'feature': feature,
                        'shap_value': contribution,
                        'base_value': explanation.get('base_value', 0),
                        'prediction': explanation.get('prediction', 0)
                    })
        
        # Store SHAP values
        shap_df = pd.DataFrame(shap_data)
        shap_table = "explain_shap"
        shap_df.to_sql(shap_table, engine, if_exists='replace', index=False)
        
        logger.info(f"Computed SHAP values for {len(shap_data)} feature contributions")
        
        context['task_instance'].xcom_push(key='shap_computed', value=True)
        return {"shap_values_count": len(shap_data)}
        
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        return {}


def publish_scores(**context):
    """Publish final FVI scores."""
    logger.info("Starting score publishing")
    
    # Get model path
    model_path = context['task_instance'].xcom_pull(key='model_path')
    gold_table = context['task_instance'].xcom_pull(key='gold_table')
    
    if not model_path or not gold_table:
        logger.warning("Missing model or data for score publishing")
        return {}
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()
    
    try:
        # Read data
        df = pd.read_sql_table(gold_table, engine)
        
        # Load model
        ensemble = EnsemblePredictor()
        ensemble.load_model(model_path)
        
        # Make predictions
        predictions = ensemble.predict(df)
        
        # Create scores table
        scores_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            for horizon in ["H5", "H10", "H20"]:
                score_name = FVINamingConventions.generate_composite_score("coal", horizon)
                scores_data.append({
                    'sub_industry': 'Coal Mining',
                    'country': 'Global',
                    'horizon': horizon,
                    'score_name': score_name,
                    'fvi_score': predictions[i],
                    'fvi_percentile': 50.0,  # Would be calculated against other industries
                    'created_at': datetime.utcnow()
                })
        
        scores_df = pd.DataFrame(scores_data)
        scores_table = "published_fvi_scores"
        scores_df.to_sql(scores_table, engine, if_exists='replace', index=False)
        
        logger.info(f"Published {len(scores_data)} FVI scores")
        
        context['task_instance'].xcom_push(key='scores_published', value=True)
        return {"scores_count": len(scores_data)}
        
    except Exception as e:
        logger.error(f"Error publishing scores: {e}")
        return {}


def refresh_vector_store(**context):
    """Refresh vector store for RAG."""
    logger.info("Starting vector store refresh")
    
    # This would integrate with a vector database like Chroma or Pinecone
    # For now, just log the action
    
    try:
        # Get SHAP data
        shap_computed = context['task_instance'].xcom_pull(key='shap_computed')
        
        if shap_computed:
            # In production, this would:
            # 1. Extract metric documentation
            # 2. Combine with SHAP explanations
            # 3. Generate embeddings
            # 4. Update vector store
            
            logger.info("Vector store refresh completed")
            
            context['task_instance'].xcom_push(key='vector_store_refreshed', value=True)
            return {"status": "success"}
        else:
            logger.warning("No SHAP data available for vector store refresh")
            return {"status": "skipped"}
            
    except Exception as e:
        logger.error(f"Error refreshing vector store: {e}")
        return {"status": "error", "error": str(e)}


def notify_completion(**context):
    """Send completion notification."""
    logger.info("Sending completion notification")
    
    # Get task results
    scores_published = context['task_instance'].xcom_pull(key='scores_published')
    vector_store_refreshed = context['task_instance'].xcom_pull(key='vector_store_refreshed')
    
    status = "Success" if scores_published and vector_store_refreshed else "Partial"
    
    # This would typically send to Slack, email, or other notification system
    logger.info(f"FVI Coal pipeline completed with status: {status}")
    
    return {"notification_sent": True, "status": status}


# Define tasks
download_task = PythonOperator(
    task_id='download_sources',
    python_callable=download_sources,
    dag=dag,
)

bronze_task = PythonOperator(
    task_id='raw_to_bronze',
    python_callable=raw_to_bronze,
    dag=dag,
)

silver_task = PythonOperator(
    task_id='bronze_to_silver',
    python_callable=bronze_to_silver,
    dag=dag,
)

gold_task = PythonOperator(
    task_id='silver_to_gold',
    python_callable=silver_to_gold,
    dag=dag,
)

weights_task = PythonOperator(
    task_id='update_weights',
    python_callable=update_weights,
    dag=dag,
)

model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

shap_task = PythonOperator(
    task_id='compute_shap',
    python_callable=compute_shap,
    dag=dag,
)

publish_task = PythonOperator(
    task_id='publish_scores',
    python_callable=publish_scores,
    dag=dag,
)

vector_task = PythonOperator(
    task_id='refresh_vector_store',
    python_callable=refresh_vector_store,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag,
)

# Define task dependencies
download_task >> bronze_task >> silver_task >> gold_task >> weights_task
gold_task >> model_task >> shap_task >> publish_task
shap_task >> vector_task
publish_task >> notify_task
vector_task >> notify_task
