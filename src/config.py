import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Database
    DATABASE_URL: str = "postgresql://fvi_user:fvi_password@localhost:5432/fvi_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    
    # Data Sources
    IEA_API_KEY: Optional[str] = None
    EIA_API_KEY: Optional[str] = None
    QUANDL_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: Optional[str] = None
    
    # Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    # ML
    MODEL_CACHE_TTL: int = 3600  # 1 hour
    MAX_RETRIES: int = 3
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    
    @validator('DATA_DIR', 'MODELS_DIR', 'LOGS_DIR')
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings