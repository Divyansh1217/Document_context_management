from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
from config import settings

if settings.DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set in the environment or .env file")

DB_SCHEMA = settings.DB_SCHEMA

engine = create_async_engine(
    str(settings.DATABASE_URL),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=False,
    connect_args={
        "statement_cache_size": 0,
        "server_settings": {
            "search_path": f"{DB_SCHEMA},public"
        },
    }
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

metadata_obj = MetaData(schema=DB_SCHEMA)
Base = declarative_base(metadata=metadata_obj)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise