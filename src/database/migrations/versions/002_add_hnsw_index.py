"""add HNSW index on vehicles.embedding_centroid

Revision ID: 002
Revises: 001
Create Date: 2026-02-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_vehicles_embedding_hnsw "
        "ON vehicles USING hnsw (embedding_centroid vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.drop_index("idx_vehicles_embedding_hnsw", table_name="vehicles")
