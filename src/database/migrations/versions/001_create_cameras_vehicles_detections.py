"""create cameras vehicles detections tables

Revision ID: 001
Revises:
Create Date: 2026-02-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "cameras",
        sa.Column("camera_id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("latitude", sa.Float, nullable=False),
        sa.Column("longitude", sa.Float, nullable=False),
        sa.Column("stream_url", sa.String(500), nullable=False),
        sa.Column("resolution", sa.String(20), nullable=True),
        sa.Column("fps", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("coverage_zone", sa.dialects.postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "vehicles",
        sa.Column(
            "vehicle_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("plate_number", sa.String(20), unique=True, nullable=True),
        sa.Column("vehicle_class", sa.String(20), nullable=True),
        sa.Column("color", sa.String(30), nullable=True),
        sa.Column("first_seen", sa.DateTime, nullable=False),
        sa.Column("last_seen", sa.DateTime, nullable=False),
        sa.Column("total_sightings", sa.Integer, server_default="1"),
        sa.Column("embedding_centroid", Vector(768), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB, nullable=True),
    )

    op.create_table(
        "detections",
        sa.Column(
            "detection_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("camera_id", sa.String(50), nullable=False),
        sa.Column(
            "vehicle_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("frame_number", sa.Integer, nullable=True),
        sa.Column("bbox_x", sa.Float, nullable=True),
        sa.Column("bbox_y", sa.Float, nullable=True),
        sa.Column("bbox_w", sa.Float, nullable=True),
        sa.Column("bbox_h", sa.Float, nullable=True),
        sa.Column("vehicle_class", sa.String(20), nullable=True),
        sa.Column("vehicle_confidence", sa.Float, nullable=True),
        sa.Column("plate_text", sa.String(20), nullable=True),
        sa.Column("plate_confidence", sa.Float, nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("track_id", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index("idx_detections_camera_time", "detections", ["camera_id", "timestamp"])
    op.create_index("idx_detections_vehicle", "detections", ["vehicle_id"])
    op.create_index("idx_detections_plate", "detections", ["plate_text"])

    op.create_foreign_key(
        "fk_detections_camera_id", "detections", "cameras", ["camera_id"], ["camera_id"]
    )
    op.create_foreign_key(
        "fk_detections_vehicle_id", "detections", "vehicles", ["vehicle_id"], ["vehicle_id"]
    )


def downgrade() -> None:
    op.drop_constraint("fk_detections_vehicle_id", "detections", type_="foreignkey")
    op.drop_constraint("fk_detections_camera_id", "detections", type_="foreignkey")
    op.drop_index("idx_detections_plate", table_name="detections")
    op.drop_index("idx_detections_vehicle", table_name="detections")
    op.drop_index("idx_detections_camera_time", table_name="detections")
    op.drop_table("detections")
    op.drop_table("vehicles")
    op.drop_table("cameras")
