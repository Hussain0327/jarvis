"""SQLAlchemy 2.0 ORM models: Camera, Vehicle, Detection with pgvector support."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    stream_url: Mapped[str] = mapped_column(String(500), nullable=False)
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True)
    fps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")
    coverage_zone: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    detections: Mapped[list[Detection]] = relationship(back_populates="camera")

    def __repr__(self) -> str:
        return f"<Camera {self.camera_id!r} ({self.name})>"


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    plate_number: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True)
    vehicle_class: Mapped[str | None] = mapped_column(String(20), nullable=True)
    color: Mapped[str | None] = mapped_column(String(30), nullable=True)
    first_seen: Mapped[datetime] = mapped_column(nullable=False)
    last_seen: Mapped[datetime] = mapped_column(nullable=False)
    total_sightings: Mapped[int] = mapped_column(Integer, default=1)
    embedding_centroid = mapped_column(Vector(768), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    detections: Mapped[list[Detection]] = relationship(back_populates="vehicle")

    def __repr__(self) -> str:
        return f"<Vehicle {self.vehicle_id!s:.8} plate={self.plate_number!r}>"


class Detection(Base):
    __tablename__ = "detections"

    detection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    camera_id: Mapped[str] = mapped_column(String(50), nullable=False)
    vehicle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_w: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_h: Mapped[float | None] = mapped_column(Float, nullable=True)
    vehicle_class: Mapped[str | None] = mapped_column(String(20), nullable=True)
    vehicle_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_text: Mapped[str | None] = mapped_column(String(20), nullable=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding = mapped_column(Vector(768), nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())

    camera: Mapped[Camera] = relationship(back_populates="detections", foreign_keys=[camera_id])
    vehicle: Mapped[Vehicle | None] = relationship(
        back_populates="detections", foreign_keys=[vehicle_id]
    )

    __table_args__ = (
        Index("idx_detections_camera_time", "camera_id", "timestamp"),
        Index("idx_detections_vehicle", "vehicle_id"),
        Index("idx_detections_plate", "plate_text"),
        {
            "comment": (
                "ForeignKeyConstraint added via migration since pgvector HNSW index "
                "requires the extension to exist first."
            )
        },
    )

    def __repr__(self) -> str:
        return f"<Detection {self.detection_id!s:.8} cam={self.camera_id}>"


# NOTE: Additional tables (trajectories, anomalies, route_clusters, predictions,
# knowledge_base, llm_query_logs) will be added via Alembic migrations in their
# respective implementation phases.
