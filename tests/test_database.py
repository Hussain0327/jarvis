"""Tests for database models and CRUD operations.

These tests require a running PostgreSQL instance with pgvector.
Run with: pytest -m integration tests/test_database.py
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from src.database.models import Camera, Detection, Vehicle

pytestmark = pytest.mark.integration


def test_create_camera(db_session, sample_camera):
    """Insert and retrieve a Camera."""
    db_session.add(sample_camera)
    db_session.flush()

    result = db_session.get(Camera, sample_camera.camera_id)
    assert result is not None
    assert result.name == "Broadway & 42nd"
    assert result.latitude == pytest.approx(40.758, abs=1e-3)


def test_create_vehicle(db_session, sample_vehicle):
    """Insert and retrieve a Vehicle."""
    db_session.add(sample_vehicle)
    db_session.flush()

    result = db_session.get(Vehicle, sample_vehicle.vehicle_id)
    assert result is not None
    assert result.plate_number == "ABC-1234"
    assert result.vehicle_class == "car"


def test_create_detection_with_relationships(db_session, sample_camera, sample_vehicle):
    """Insert a Detection linked to Camera and Vehicle."""
    db_session.add(sample_camera)
    db_session.add(sample_vehicle)
    db_session.flush()

    detection = Detection(
        detection_id=uuid.uuid4(),
        camera_id=sample_camera.camera_id,
        vehicle_id=sample_vehicle.vehicle_id,
        timestamp=datetime.now(UTC),
        frame_number=42,
        bbox_x=100.0,
        bbox_y=200.0,
        bbox_w=50.0,
        bbox_h=30.0,
        vehicle_class="car",
        vehicle_confidence=0.95,
        plate_text="ABC-1234",
        plate_confidence=0.88,
        track_id=7,
    )
    db_session.add(detection)
    db_session.flush()

    result = db_session.get(Detection, detection.detection_id)
    assert result is not None
    assert result.camera.camera_id == "cam-001"
    assert result.vehicle.plate_number == "ABC-1234"


def test_camera_detections_relationship(db_session, sample_camera, sample_vehicle):
    """Camera.detections back-populates correctly."""
    db_session.add(sample_camera)
    db_session.add(sample_vehicle)
    db_session.flush()

    for i in range(3):
        db_session.add(
            Detection(
                detection_id=uuid.uuid4(),
                camera_id=sample_camera.camera_id,
                vehicle_id=sample_vehicle.vehicle_id,
                timestamp=datetime.now(UTC),
                frame_number=i,
                vehicle_class="car",
                vehicle_confidence=0.9,
            )
        )
    db_session.flush()

    camera = db_session.get(Camera, sample_camera.camera_id)
    assert len(camera.detections) == 3
