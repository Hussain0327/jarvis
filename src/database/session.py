"""Database engine and session management."""

from __future__ import annotations

import threading

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
_lock = threading.Lock()


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine (created on first call)."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                settings = get_settings()
                _engine = create_engine(
                    settings.database.url,
                    pool_size=settings.database.pool_size,
                    pool_pre_ping=True,
                    echo=settings.database.echo,
                )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the shared session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()  # resolve outside _lock to avoid deadlock
        with _lock:
            if _SessionLocal is None:
                _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionLocal


def get_session() -> Session:
    """Create and return a new database session."""
    return get_session_factory()()


def ensure_pgvector_extension(engine: Engine | None = None) -> None:
    """Create the pgvector extension if it doesn't exist."""
    eng = engine or get_engine()
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def reset_engine() -> None:
    """Dispose of the current engine. Useful for testing."""
    global _engine, _SessionLocal
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None
