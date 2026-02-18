"""Alembic migration environment.

Auto-creates the pgvector extension before running migrations.
"""

from alembic import context
from sqlalchemy import create_engine, pool, text

from src.database.models import Base

config = context.config
target_metadata = Base.metadata


def _ensure_pgvector(connection) -> None:
    """Create pgvector extension if it doesn't exist."""
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.commit()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without connecting)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connectable = create_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _ensure_pgvector(connection)

        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
