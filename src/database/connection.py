from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis
from typing import Generator
import logging

from config import settings

logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis connection
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis_client


def init_db() -> None:
    """Initialize database tables."""
    from ..models.database import Base
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def test_db_connection() -> bool:
    """Test database connection."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def test_redis_connection() -> bool:
    """Test Redis connection."""
    try:
        redis_client.ping()
        logger.info("Redis connection test successful")
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.redis_client = redis_client
    
    def create_tables(self):
        """Create all database tables."""
        from ..models.database import Base
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        from ..models.database import Base
        Base.metadata.drop_all(bind=self.engine)
    
    def reset_database(self):
        """Reset database (drop and recreate tables)."""
        self.drop_tables()
        self.create_tables()
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: dict = None):
        """Execute raw SQL query."""
        with self.engine.connect() as conn:
            if params:
                return conn.execute(sql, params)
            else:
                return conn.execute(sql)


# Global database manager instance
db_manager = DatabaseManager()
