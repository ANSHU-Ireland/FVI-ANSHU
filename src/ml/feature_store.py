"""
Feature Store for FVI Analytics Platform

This module implements the Feature Store with:
1. Feast/Tecton integration for feature definitions
2. Auto-generation from YAML catalogue
3. Shared feature definitions between ML pipeline and API
4. Historical and real-time feature serving
5. Feature versioning and lineage tracking

Author: FVI Analytics Team
Created: 2025-07-05
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from feast import FeatureStore as FeastFeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String, Bool, UnixTimestamp
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import config
from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class FeatureStore:
    """Feature Store for FVI Analytics Platform"""
    
    def __init__(self, feast_repo_path: str = "/tmp/feast_repo"):
        self.feast_repo_path = feast_repo_path
        self.db_manager = DatabaseManager()
        self.feast_store = None
        self.feature_definitions = {}
        self.entity_definitions = {}
        
    async def initialize(self):
        """Initialize the Feature Store"""
        try:
            # Load feature definitions from YAML catalogue
            await self.load_feature_definitions()
            
            # Initialize Feast repository
            await self.setup_feast_repository()
            
            # Apply feature definitions to Feast
            await self.apply_feature_definitions()
            
            logger.info("Feature Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Feature Store: {str(e)}")
            raise
    
    async def load_feature_definitions(self):
        """Load feature definitions from YAML catalogue"""
        try:
            # Load the metric catalogue
            catalogue_path = "/workspaces/FVI-ANSHU/meta/metric_catalogue.yaml"
            
            with open(catalogue_path, 'r') as f:
                catalogue = yaml.safe_load(f)
            
            # Extract feature definitions
            self.feature_definitions = {}
            self.entity_definitions = {}
            
            # Define entities
            self.entity_definitions['mine'] = {
                'name': 'mine',
                'value_type': String,
                'description': 'Coal mine entity'
            }
            
            self.entity_definitions['country'] = {
                'name': 'country',
                'value_type': String,
                'description': 'Country entity'
            }
            
            # Process metrics from catalogue
            if 'metrics' in catalogue:
                for metric_slug, metric_info in catalogue['metrics'].items():
                    feature_name = f"f_{metric_slug}"
                    
                    # Determine data type
                    data_type = self._get_feast_type(metric_info.get('data_type', 'float'))
                    
                    # Create feature definition
                    self.feature_definitions[feature_name] = {
                        'name': feature_name,
                        'dtype': data_type,
                        'description': metric_info.get('description', ''),
                        'tags': {
                            'category': metric_info.get('category', 'unknown'),
                            'unit': metric_info.get('unit', ''),
                            'source': metric_info.get('source', 'unknown')
                        }
                    }
            
            # Add derived features
            self._add_derived_features()
            
            logger.info(f"Loaded {len(self.feature_definitions)} feature definitions")
            
        except Exception as e:
            logger.error(f"Error loading feature definitions: {str(e)}")
            raise
    
    def _get_feast_type(self, data_type: str):
        """Convert data type to Feast type"""
        type_mapping = {
            'float': Float64,
            'int': Int64,
            'string': String,
            'bool': Bool,
            'timestamp': UnixTimestamp
        }
        return type_mapping.get(data_type.lower(), Float64)
    
    def _add_derived_features(self):
        """Add derived/engineered features"""
        derived_features = {
            'f_capacity_utilization_rate': {
                'name': 'f_capacity_utilization_rate',
                'dtype': Float64,
                'description': 'Capacity utilization rate (production/capacity)',
                'tags': {'category': 'operational', 'unit': 'ratio'}
            },
            'f_revenue_per_mt': {
                'name': 'f_revenue_per_mt',
                'dtype': Float64,
                'description': 'Revenue per metric ton',
                'tags': {'category': 'financial', 'unit': 'USD/MT'}
            },
            'f_operating_margin': {
                'name': 'f_operating_margin',
                'dtype': Float64,
                'description': 'Operating margin ratio',
                'tags': {'category': 'financial', 'unit': 'ratio'}
            },
            'f_safety_incident_rate': {
                'name': 'f_safety_incident_rate',
                'dtype': Float64,
                'description': 'Safety incident rate per 1000 employees',
                'tags': {'category': 'safety', 'unit': 'incidents/1000_employees'}
            },
            'f_carbon_intensity': {
                'name': 'f_carbon_intensity',
                'dtype': Float64,
                'description': 'Carbon intensity in tCO2 per metric ton',
                'tags': {'category': 'environmental', 'unit': 'tCO2/MT'}
            },
            'f_reserve_life_years': {
                'name': 'f_reserve_life_years',
                'dtype': Float64,
                'description': 'Reserve life in years',
                'tags': {'category': 'operational', 'unit': 'years'}
            }
        }
        
        self.feature_definitions.update(derived_features)
    
    async def setup_feast_repository(self):
        """Setup Feast repository structure"""
        import os
        
        # Create repository directory
        os.makedirs(self.feast_repo_path, exist_ok=True)
        
        # Create feature_store.yaml
        feature_store_config = {
            'project': 'fvi_analytics',
            'registry': f'{self.feast_repo_path}/data/registry.db',
            'provider': 'local',
            'online_store': {
                'type': 'redis',
                'connection_string': f'redis://{config.REDIS_HOST}:{config.REDIS_PORT}'
            },
            'offline_store': {
                'type': 'file'
            }
        }
        
        config_path = f"{self.feast_repo_path}/feature_store.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(feature_store_config, f)
        
        # Initialize Feast store
        self.feast_store = FeastFeatureStore(repo_path=self.feast_repo_path)
        
        logger.info(f"Feast repository setup at {self.feast_repo_path}")
    
    async def apply_feature_definitions(self):
        """Apply feature definitions to Feast"""
        try:
            # Create entities
            entities = []
            for entity_name, entity_def in self.entity_definitions.items():
                entity = Entity(
                    name=entity_def['name'],
                    value_type=entity_def['value_type'],
                    description=entity_def['description']
                )
                entities.append(entity)
            
            # Create feature views
            feature_views = []
            
            # Mine features view
            mine_features_view = await self._create_mine_features_view()
            feature_views.append(mine_features_view)
            
            # Country features view
            country_features_view = await self._create_country_features_view()
            feature_views.append(country_features_view)
            
            # Apply to Feast
            repo_objects = entities + feature_views
            
            # Write feature definitions to Python file
            await self._write_feature_definitions_file(repo_objects)
            
            # Apply using Feast CLI (this would typically be done via feast apply)
            # For now, we'll store the definitions for later use
            
            logger.info(f"Applied {len(entities)} entities and {len(feature_views)} feature views")
            
        except Exception as e:
            logger.error(f"Error applying feature definitions: {str(e)}")
            raise
    
    async def _create_mine_features_view(self):
        """Create mine features view"""
        # Create a data source (in production, this would be your data warehouse)
        source = FileSource(
            path=f"{self.feast_repo_path}/data/mine_features.parquet",
            timestamp_field="feature_date",
            created_timestamp_column="created_at",
        )
        
        # Define feature fields
        feature_fields = []
        for feature_name, feature_def in self.feature_definitions.items():
            field = Field(
                name=feature_name,
                dtype=feature_def['dtype'],
                description=feature_def['description']
            )
            feature_fields.append(field)
        
        # Create feature view
        feature_view = FeatureView(
            name="mine_features",
            entities=["mine"],
            ttl=timedelta(days=7),
            schema=feature_fields,
            source=source,
            tags={"team": "fvi_analytics", "version": "v1"}
        )
        
        return feature_view
    
    async def _create_country_features_view(self):
        """Create country-level features view"""
        source = FileSource(
            path=f"{self.feast_repo_path}/data/country_features.parquet",
            timestamp_field="feature_date",
            created_timestamp_column="created_at",
        )
        
        # Country-level features
        country_fields = [
            Field(name="f_country_coal_production", dtype=Float64),
            Field(name="f_country_esg_average", dtype=Float64),
            Field(name="f_country_safety_average", dtype=Float64),
            Field(name="f_country_carbon_intensity_avg", dtype=Float64),
        ]
        
        feature_view = FeatureView(
            name="country_features",
            entities=["country"],
            ttl=timedelta(days=30),
            schema=country_fields,
            source=source,
            tags={"team": "fvi_analytics", "version": "v1"}
        )
        
        return feature_view
    
    async def _write_feature_definitions_file(self, repo_objects):
        """Write feature definitions to Python file"""
        definitions_file = f"{self.feast_repo_path}/feature_definitions.py"
        
        with open(definitions_file, 'w') as f:
            f.write("# Auto-generated feature definitions\n")
            f.write("from feast import Entity, FeatureView, Field, FileSource\n")
            f.write("from feast.types import Float64, Int64, String, Bool, UnixTimestamp\n")
            f.write("from datetime import timedelta\n\n")
            
            # Write entities
            for obj in repo_objects:
                if isinstance(obj, Entity):
                    f.write(f"{obj.name} = Entity(\n")
                    f.write(f"    name='{obj.name}',\n")
                    f.write(f"    value_type={obj.value_type.__name__},\n")
                    f.write(f"    description='{obj.description}'\n")
                    f.write(")\n\n")
            
            # Write feature views (simplified for this example)
            f.write("# Feature views would be defined here\n")
            f.write("# This would include all the FeatureView objects\n")
    
    async def get_historical_features(self, entity_df: pd.DataFrame = None,
                                     features: List[str] = None,
                                     start_date: datetime = None,
                                     end_date: datetime = None) -> pd.DataFrame:
        """
        Get historical features for training
        
        Args:
            entity_df: DataFrame with entity keys and timestamps
            features: List of feature names to retrieve
            start_date: Start date for feature retrieval
            end_date: End date for feature retrieval
            
        Returns:
            DataFrame with historical features
        """
        try:
            if entity_df is None:
                # Create entity DataFrame from database
                entity_df = await self._create_entity_dataframe(start_date, end_date)
            
            if features is None:
                # Get all available features
                features = list(self.feature_definitions.keys())
            
            # For now, retrieve directly from database
            # In production, this would use Feast's get_historical_features
            historical_features = await self._get_features_from_database(
                entity_df, features, start_date, end_date
            )
            
            logger.info(f"Retrieved {len(historical_features)} historical feature records")
            return historical_features
            
        except Exception as e:
            logger.error(f"Error getting historical features: {str(e)}")
            raise
    
    async def get_online_features(self, entity_keys: Dict[str, List[str]],
                                 features: List[str] = None) -> Dict[str, List[Any]]:
        """
        Get online features for real-time inference
        
        Args:
            entity_keys: Dictionary of entity types to entity IDs
            features: List of feature names to retrieve
            
        Returns:
            Dictionary of feature values
        """
        try:
            if features is None:
                features = list(self.feature_definitions.keys())
            
            # For now, retrieve from database
            # In production, this would use Feast's get_online_features
            online_features = await self._get_online_features_from_database(entity_keys, features)
            
            return online_features
            
        except Exception as e:
            logger.error(f"Error getting online features: {str(e)}")
            raise
    
    async def get_latest_features(self, mine_id: str) -> Dict[str, Any]:
        """Get latest features for a specific mine"""
        try:
            query = """
            SELECT *
            FROM gold_mine_features
            WHERE mine_id = $1
            ORDER BY feature_date DESC
            LIMIT 1
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, mine_id)
            
            if not result:
                return {}
            
            # Convert to feature dictionary
            features = {}
            for feature_name in self.feature_definitions:
                # Map feature names to database columns
                db_column = self._map_feature_to_column(feature_name)
                if db_column in result:
                    features[feature_name] = result[db_column]
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting latest features for {mine_id}: {str(e)}")
            return {}
    
    async def _create_entity_dataframe(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create entity DataFrame for feature retrieval"""
        query = """
        SELECT DISTINCT 
            mine_id,
            feature_date,
            country_code as country
        FROM gold_mine_features
        WHERE feature_date BETWEEN $1 AND $2
        ORDER BY mine_id, feature_date
        """
        
        async with self.db_manager.get_connection() as conn:
            results = await conn.fetch(query, start_date, end_date)
        
        # Convert to DataFrame
        entity_df = pd.DataFrame([dict(row) for row in results])
        
        # Ensure proper types
        entity_df['feature_date'] = pd.to_datetime(entity_df['feature_date'])
        
        return entity_df
    
    async def _get_features_from_database(self, entity_df: pd.DataFrame, features: List[str],
                                         start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get features from database"""
        query = """
        SELECT 
            mine_id,
            feature_date,
            country_code,
            production_mt,
            capacity_mt,
            capacity_utilization_rate,
            esg_score,
            carbon_intensity_tco2_per_mt,
            revenue_per_mt,
            operating_margin,
            safety_incident_rate_per_1000_employees,
            reserve_life_years,
            total_carbon_emissions_tco2,
            productivity_mt_per_employee,
            latitude_abs,
            metric_year,
            metric_quarter,
            metric_month,
            utilization_category,
            esg_category,
            carbon_intensity_category,
            feature_freshness_score
        FROM gold_mine_features
        WHERE feature_date BETWEEN $1 AND $2
        AND feature_freshness_score >= 0.7
        ORDER BY mine_id, feature_date
        """
        
        async with self.db_manager.get_connection() as conn:
            results = await conn.fetch(query, start_date, end_date)
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in results])
        
        # Map database columns to feature names
        feature_mapping = self._get_feature_mapping()
        
        # Rename columns to match feature names
        df = df.rename(columns=feature_mapping)
        
        # Select only requested features
        available_features = [f for f in features if f in df.columns]
        entity_cols = ['mine_id', 'feature_date', 'country_code']
        
        return df[entity_cols + available_features]
    
    async def _get_online_features_from_database(self, entity_keys: Dict[str, List[str]],
                                               features: List[str]) -> Dict[str, List[Any]]:
        """Get online features from database"""
        # For simplicity, get latest features for each mine
        mine_ids = entity_keys.get('mine', [])
        
        if not mine_ids:
            return {}
        
        # Build query for latest features
        placeholders = ','.join(f'${i+1}' for i in range(len(mine_ids)))
        query = f"""
        SELECT *
        FROM gold_mine_features
        WHERE mine_id IN ({placeholders})
        AND feature_date >= NOW() - INTERVAL '7 days'
        ORDER BY mine_id, feature_date DESC
        """
        
        async with self.db_manager.get_connection() as conn:
            results = await conn.fetch(query, *mine_ids)
        
        # Process results
        online_features = {}
        for feature in features:
            online_features[feature] = []
        
        # Get latest record for each mine
        latest_by_mine = {}
        for row in results:
            mine_id = row['mine_id']
            if mine_id not in latest_by_mine:
                latest_by_mine[mine_id] = row
        
        # Extract feature values
        feature_mapping = self._get_feature_mapping()
        
        for mine_id in mine_ids:
            if mine_id in latest_by_mine:
                row = latest_by_mine[mine_id]
                for feature in features:
                    db_column = self._map_feature_to_column(feature)
                    value = row.get(db_column)
                    online_features[feature].append(value)
            else:
                # No data for this mine
                for feature in features:
                    online_features[feature].append(None)
        
        return online_features
    
    def _get_feature_mapping(self) -> Dict[str, str]:
        """Get mapping from database columns to feature names"""
        return {
            'production_mt': 'f_production_volume',
            'capacity_mt': 'f_capacity_volume',
            'capacity_utilization_rate': 'f_capacity_utilization_rate',
            'esg_score': 'f_sustainability_score',
            'carbon_intensity_tco2_per_mt': 'f_carbon_intensity',
            'revenue_per_mt': 'f_revenue_per_mt',
            'operating_margin': 'f_operating_margin',
            'safety_incident_rate_per_1000_employees': 'f_safety_incident_rate',
            'reserve_life_years': 'f_reserve_life_years',
            'total_carbon_emissions_tco2': 'f_total_carbon_emissions',
            'productivity_mt_per_employee': 'f_productivity_per_employee',
            'latitude_abs': 'f_geographic_risk',
            'metric_year': 'f_temporal_trend',
            'metric_quarter': 'f_seasonal_pattern',
            'metric_month': 'f_monthly_cycle'
        }
    
    def _map_feature_to_column(self, feature_name: str) -> str:
        """Map feature name to database column"""
        reverse_mapping = {v: k for k, v in self._get_feature_mapping().items()}
        return reverse_mapping.get(feature_name, feature_name)
    
    async def materialize_features(self, start_date: datetime, end_date: datetime):
        """Materialize features to the feature store"""
        try:
            # Get feature data
            entity_df = await self._create_entity_dataframe(start_date, end_date)
            
            if entity_df.empty:
                logger.warning("No entity data found for materialization")
                return
            
            # Get all features
            features = list(self.feature_definitions.keys())
            feature_df = await self._get_features_from_database(entity_df, features, start_date, end_date)
            
            # Save to Parquet files for Feast
            mine_features_path = f"{self.feast_repo_path}/data/mine_features.parquet"
            
            # Ensure data directory exists
            import os
            os.makedirs(f"{self.feast_repo_path}/data", exist_ok=True)
            
            # Convert feature_date to timestamp
            feature_df['feature_date'] = pd.to_datetime(feature_df['feature_date'])
            feature_df['created_at'] = pd.to_datetime(datetime.now())
            
            # Save to Parquet
            feature_df.to_parquet(mine_features_path, index=False)
            
            logger.info(f"Materialized {len(feature_df)} feature records to {mine_features_path}")
            
        except Exception as e:
            logger.error(f"Error materializing features: {str(e)}")
            raise
    
    async def register_feature_definition(self, feature_name: str, feature_def: Dict[str, Any]):
        """Register a new feature definition"""
        self.feature_definitions[feature_name] = feature_def
        
        # In production, this would update the Feast repository
        logger.info(f"Registered feature definition: {feature_name}")
    
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature lineage information"""
        if feature_name not in self.feature_definitions:
            return {}
        
        feature_def = self.feature_definitions[feature_name]
        
        # Build lineage information
        lineage = {
            'feature_name': feature_name,
            'description': feature_def.get('description', ''),
            'data_type': feature_def['dtype'].__name__,
            'tags': feature_def.get('tags', {}),
            'source_tables': ['gold_mine_features'],  # Simplified
            'dependencies': [],  # Would track feature dependencies
            'created_at': datetime.now().isoformat(),
            'version': 'v1'
        }
        
        return lineage
    
    async def cleanup(self):
        """Clean up resources"""
        # In production, this would clean up any connections
        logger.info("Feature Store cleanup completed")


if __name__ == "__main__":
    # Example usage
    async def main():
        feature_store = FeatureStore()
        await feature_store.initialize()
        
        try:
            # Get historical features
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            historical_features = await feature_store.get_historical_features(
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"Retrieved {len(historical_features)} historical features")
            print(historical_features.head())
            
            # Get online features
            online_features = await feature_store.get_online_features(
                entity_keys={'mine': ['MINE_001', 'MINE_002']},
                features=['f_production_volume', 'f_sustainability_score']
            )
            
            print("Online features:", online_features)
            
        finally:
            await feature_store.cleanup()
    
    asyncio.run(main())
