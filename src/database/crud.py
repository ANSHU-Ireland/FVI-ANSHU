from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import logging

from models.database import DataSource, MetricDefinition, MetricValue, CompositeScore, ModelRun, ChatSession, ChatMessage
from models.schemas import (
    DataSourceCreate, MetricDefinitionCreate, MetricValueCreate, 
    CompositeScoreCreate, ModelRunCreate, ChatSessionCreate, ChatMessageCreate
)

logger = logging.getLogger(__name__)


class DataSourceCRUD:
    """CRUD operations for DataSource model."""
    
    @staticmethod
    def create(db: Session, data_source: DataSourceCreate) -> DataSource:
        """Create a new data source."""
        db_data_source = DataSource(**data_source.dict())
        db.add(db_data_source)
        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    
    @staticmethod
    def get(db: Session, data_source_id: int) -> Optional[DataSource]:
        """Get data source by ID."""
        return db.query(DataSource).filter(DataSource.id == data_source_id).first()
    
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[DataSource]:
        """Get data source by name."""
        return db.query(DataSource).filter(DataSource.name == name).first()
    
    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> List[DataSource]:
        """Get multiple data sources."""
        return db.query(DataSource).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, data_source_id: int, data_source: DataSourceCreate) -> Optional[DataSource]:
        """Update data source."""
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source:
            for key, value in data_source.dict().items():
                setattr(db_data_source, key, value)
            db.commit()
            db.refresh(db_data_source)
        return db_data_source
    
    @staticmethod
    def delete(db: Session, data_source_id: int) -> bool:
        """Delete data source."""
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source:
            db.delete(db_data_source)
            db.commit()
            return True
        return False


class MetricDefinitionCRUD:
    """CRUD operations for MetricDefinition model."""
    
    @staticmethod
    def create(db: Session, metric: MetricDefinitionCreate) -> MetricDefinition:
        """Create a new metric definition."""
        db_metric = MetricDefinition(**metric.dict())
        # Calculate confidence score based on data quality flags
        db_metric.final_data_confidence_score = MetricDefinitionCRUD._calculate_confidence_score(db_metric)
        db.add(db_metric)
        db.commit()
        db.refresh(db_metric)
        return db_metric
    
    @staticmethod
    def get(db: Session, metric_id: int) -> Optional[MetricDefinition]:
        """Get metric definition by ID."""
        return db.query(MetricDefinition).filter(MetricDefinition.id == metric_id).first()
    
    @staticmethod
    def get_by_slug(db: Session, slug: str) -> Optional[MetricDefinition]:
        """Get metric definition by slug."""
        return db.query(MetricDefinition).filter(MetricDefinition.metric_slug == slug).first()
    
    @staticmethod
    def get_by_sheet(db: Session, sheet_name: str) -> List[MetricDefinition]:
        """Get metric definitions by sheet name."""
        return db.query(MetricDefinition).filter(MetricDefinition.sheet_name == sheet_name).all()
    
    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> List[MetricDefinition]:
        """Get multiple metric definitions."""
        return db.query(MetricDefinition).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, metric_id: int, metric: MetricDefinitionCreate) -> Optional[MetricDefinition]:
        """Update metric definition."""
        db_metric = db.query(MetricDefinition).filter(MetricDefinition.id == metric_id).first()
        if db_metric:
            for key, value in metric.dict().items():
                setattr(db_metric, key, value)
            # Recalculate confidence score
            db_metric.final_data_confidence_score = MetricDefinitionCRUD._calculate_confidence_score(db_metric)
            db.commit()
            db.refresh(db_metric)
        return db_metric
    
    @staticmethod
    def delete(db: Session, metric_id: int) -> bool:
        """Delete metric definition."""
        db_metric = db.query(MetricDefinition).filter(MetricDefinition.id == metric_id).first()
        if db_metric:
            db.delete(db_metric)
            db.commit()
            return True
        return False
    
    @staticmethod
    def _calculate_confidence_score(metric: MetricDefinition) -> float:
        """Calculate confidence score based on data quality flags."""
        scores = [
            metric.structured_availability or 0,
            metric.country_level_availability or 0,
            metric.sub_industry_availability or 0,
            metric.volume_of_data or 0,
            metric.alternative_proxy_feasibility or 0,
            metric.genai_ml_fillability or 0,
            metric.longitudinal_availability or 0,
            (6 - (metric.data_verification_bias_risk or 3)),  # Inverted scale
            (6 - (metric.interdependence_with_other_metrics or 3))  # Inverted scale
        ]
        
        # Filter out None values
        valid_scores = [s for s in scores if s is not None]
        
        if valid_scores:
            return sum(valid_scores) / len(valid_scores)
        return 2.5  # Default middle score


class MetricValueCRUD:
    """CRUD operations for MetricValue model."""
    
    @staticmethod
    def create(db: Session, metric_value: MetricValueCreate) -> MetricValue:
        """Create a new metric value."""
        db_metric_value = MetricValue(**metric_value.dict())
        db.add(db_metric_value)
        db.commit()
        db.refresh(db_metric_value)
        return db_metric_value
    
    @staticmethod
    def get(db: Session, metric_value_id: int) -> Optional[MetricValue]:
        """Get metric value by ID."""
        return db.query(MetricValue).filter(MetricValue.id == metric_value_id).first()
    
    @staticmethod
    def get_by_metric_and_entity(
        db: Session, 
        metric_id: int, 
        sub_industry: str,
        country: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[MetricValue]:
        """Get metric values for specific metric and entity."""
        query = db.query(MetricValue).filter(
            MetricValue.metric_definition_id == metric_id,
            MetricValue.sub_industry == sub_industry
        )
        
        if country:
            query = query.filter(MetricValue.country == country)
        if year:
            query = query.filter(MetricValue.year == year)
            
        return query.all()
    
    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> List[MetricValue]:
        """Get multiple metric values."""
        return db.query(MetricValue).offset(skip).limit(limit).all()
    
    @staticmethod
    def update(db: Session, metric_value_id: int, metric_value: MetricValueCreate) -> Optional[MetricValue]:
        """Update metric value."""
        db_metric_value = db.query(MetricValue).filter(MetricValue.id == metric_value_id).first()
        if db_metric_value:
            for key, value in metric_value.dict().items():
                setattr(db_metric_value, key, value)
            db.commit()
            db.refresh(db_metric_value)
        return db_metric_value
    
    @staticmethod
    def delete(db: Session, metric_value_id: int) -> bool:
        """Delete metric value."""
        db_metric_value = db.query(MetricValue).filter(MetricValue.id == metric_value_id).first()
        if db_metric_value:
            db.delete(db_metric_value)
            db.commit()
            return True
        return False


class CompositeScoreCRUD:
    """CRUD operations for CompositeScore model."""
    
    @staticmethod
    def create(db: Session, composite_score: CompositeScoreCreate) -> CompositeScore:
        """Create a new composite score."""
        db_composite_score = CompositeScore(**composite_score.dict())
        db.add(db_composite_score)
        db.commit()
        db.refresh(db_composite_score)
        return db_composite_score
    
    @staticmethod
    def get(db: Session, composite_score_id: int) -> Optional[CompositeScore]:
        """Get composite score by ID."""
        return db.query(CompositeScore).filter(CompositeScore.id == composite_score_id).first()
    
    @staticmethod
    def get_by_entity_and_horizon(
        db: Session, 
        sub_industry: str,
        horizon: int,
        country: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[CompositeScore]:
        """Get composite scores for specific entity and horizon."""
        query = db.query(CompositeScore).filter(
            CompositeScore.sub_industry == sub_industry,
            CompositeScore.horizon == horizon
        )
        
        if country:
            query = query.filter(CompositeScore.country == country)
        if year:
            query = query.filter(CompositeScore.year == year)
            
        return query.all()
    
    @staticmethod
    def get_latest_by_entity(
        db: Session, 
        sub_industry: str,
        country: Optional[str] = None
    ) -> List[CompositeScore]:
        """Get latest composite scores for an entity."""
        query = db.query(CompositeScore).filter(
            CompositeScore.sub_industry == sub_industry
        )
        
        if country:
            query = query.filter(CompositeScore.country == country)
            
        return query.order_by(CompositeScore.created_at.desc()).all()
    
    @staticmethod
    def get_multi(db: Session, skip: int = 0, limit: int = 100) -> List[CompositeScore]:
        """Get multiple composite scores."""
        return db.query(CompositeScore).offset(skip).limit(limit).all()


class ModelRunCRUD:
    """CRUD operations for ModelRun model."""
    
    @staticmethod
    def create(db: Session, model_run: ModelRunCreate) -> ModelRun:
        """Create a new model run."""
        db_model_run = ModelRun(**model_run.dict())
        db.add(db_model_run)
        db.commit()
        db.refresh(db_model_run)
        return db_model_run
    
    @staticmethod
    def get(db: Session, model_run_id: int) -> Optional[ModelRun]:
        """Get model run by ID."""
        return db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
    
    @staticmethod
    def get_latest_by_model_name(db: Session, model_name: str) -> Optional[ModelRun]:
        """Get latest model run by model name."""
        return db.query(ModelRun).filter(
            ModelRun.model_name == model_name
        ).order_by(ModelRun.created_at.desc()).first()
    
    @staticmethod
    def get_deployed_models(db: Session) -> List[ModelRun]:
        """Get all deployed models."""
        return db.query(ModelRun).filter(ModelRun.is_deployed == True).all()
    
    @staticmethod
    def update_deployment_status(db: Session, model_run_id: int, is_deployed: bool) -> Optional[ModelRun]:
        """Update deployment status of a model run."""
        db_model_run = db.query(ModelRun).filter(ModelRun.id == model_run_id).first()
        if db_model_run:
            db_model_run.is_deployed = is_deployed
            if is_deployed:
                from datetime import datetime
                db_model_run.deployment_date = datetime.utcnow()
            db.commit()
            db.refresh(db_model_run)
        return db_model_run


class ChatCRUD:
    """CRUD operations for Chat models."""
    
    @staticmethod
    def create_session(db: Session, session: ChatSessionCreate) -> ChatSession:
        """Create a new chat session."""
        db_session = ChatSession(**session.dict())
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session
    
    @staticmethod
    def create_message(db: Session, message: ChatMessageCreate) -> ChatMessage:
        """Create a new chat message."""
        db_message = ChatMessage(**message.dict())
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        # Update session last activity
        db_session = db.query(ChatSession).filter(
            ChatSession.session_id == message.session_id
        ).first()
        if db_session:
            from datetime import datetime
            db_session.last_activity = datetime.utcnow()
            db.commit()
        
        return db_message
    
    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        return db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    
    @staticmethod
    def get_session_messages(db: Session, session_id: str) -> List[ChatMessage]:
        """Get all messages for a chat session."""
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.asc()).all()
    
    @staticmethod
    def update_session_context(db: Session, session_id: str, context: Dict[str, Any]) -> Optional[ChatSession]:
        """Update chat session context."""
        db_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if db_session:
            db_session.context = context
            from datetime import datetime
            db_session.last_activity = datetime.utcnow()
            db.commit()
            db.refresh(db_session)
        return db_session


# Export CRUD classes
__all__ = [
    "DataSourceCRUD",
    "MetricDefinitionCRUD", 
    "MetricValueCRUD",
    "CompositeScoreCRUD",
    "ModelRunCRUD",
    "ChatCRUD",
    "FVICrudOperations"
]


class FVICrudOperations:
    """Unified CRUD operations for FVI system."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.data_source_crud = DataSourceCRUD()
        self.metric_definition_crud = MetricDefinitionCRUD()
        self.metric_value_crud = MetricValueCRUD()
        self.composite_score_crud = CompositeScoreCRUD()
        self.model_run_crud = ModelRunCRUD()
        self.chat_crud = ChatCRUD()
    
    async def create_data_source(self, data_source: DataSourceCreate):
        """Create a data source."""
        async with self.db_manager.get_session() as session:
            return self.data_source_crud.create(session, data_source)
    
    async def create_metric_definition(self, metric_def: MetricDefinitionCreate):
        """Create a metric definition."""
        async with self.db_manager.get_session() as session:
            return self.metric_definition_crud.create(session, metric_def)
    
    async def create_company(self, company):
        """Create a company (placeholder - using data source for now)."""
        # For now, create as a data source since we don't have a separate company model
        data_source = DataSourceCreate(
            name=company.name,
            source_type="company",
            description=f"Company: {company.name}",
            metadata=company.dict()
        )
        return await self.create_data_source(data_source)
    
    async def create_metric_data(self, metric_data):
        """Create metric data (using MetricValue)."""
        async with self.db_manager.get_session() as session:
            # Convert to MetricValueCreate
            metric_value = MetricValueCreate(
                metric_definition_id=metric_data.metric_definition_id,
                company_id=metric_data.company_id,
                value=metric_data.value,
                normalized_value=metric_data.normalized_value,
                period_start=metric_data.period_start,
                period_end=metric_data.period_end,
                data_source_id=metric_data.data_source_id,
                metadata=metric_data.metadata
            )
            return self.metric_value_crud.create(session, metric_value)
    
    async def create_horizon_weight(self, horizon_weight):
        """Create horizon weight (placeholder implementation)."""
        # For now, we'll store this as metadata in a data source
        data_source = DataSourceCreate(
            name=f"Weight_{horizon_weight.horizon}",
            source_type="weight",
            description=f"Horizon weight for {horizon_weight.horizon}",
            metadata=horizon_weight.dict()
        )
        return await self.create_data_source(data_source)
    
    async def get_companies(self):
        """Get companies (using data sources with type company)."""
        async with self.db_manager.get_session() as session:
            return self.data_source_crud.get_all(session, source_type="company")
    
    async def get_metric_definitions(self):
        """Get all metric definitions."""
        async with self.db_manager.get_session() as session:
            return self.metric_definition_crud.get_all(session)
