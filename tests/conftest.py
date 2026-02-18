"""Shared pytest fixtures: database engine, session with transactional rollback, samples."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from src.database.models import Base, Camera, Detection, Vehicle


@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine (requires running PostgreSQL)."""
    from src.config import get_settings

    settings = get_settings()
    engine = create_engine(settings.database.url, echo=False)

    # Ensure pgvector extension exists
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Teardown: drop all tables
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """Provide a transactional database session that rolls back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def sample_camera() -> Camera:
    return Camera(
        camera_id="cam-001",
        name="Broadway & 42nd",
        latitude=40.7580,
        longitude=-73.9855,
        stream_url="https://example.com/cam001.mjpeg",
        status="active",
    )


@pytest.fixture()
def sample_vehicle() -> Vehicle:
    return Vehicle(
        vehicle_id=uuid.uuid4(),
        plate_number="ABC-1234",
        vehicle_class="car",
        first_seen=datetime.now(UTC),
        last_seen=datetime.now(UTC),
        total_sightings=1,
    )


@pytest.fixture()
def sample_detection(sample_camera, sample_vehicle) -> Detection:
    return Detection(
        detection_id=uuid.uuid4(),
        camera_id=sample_camera.camera_id,
        vehicle_id=sample_vehicle.vehicle_id,
        timestamp=datetime.now(UTC),
        frame_number=1,
        bbox_x=100.0,
        bbox_y=200.0,
        bbox_w=50.0,
        bbox_h=30.0,
        vehicle_class="car",
        vehicle_confidence=0.92,
        plate_text="ABC-1234",
        plate_confidence=0.85,
        track_id=1,
    )
