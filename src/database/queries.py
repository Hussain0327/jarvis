"""CRUD query functions for Camera, Vehicle, and Detection.

All functions take an explicit Session parameter. Caller manages commit/rollback.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.base import DetectionResult, PerceptionResult
from src.database.models import Camera, Detection, Vehicle

# ---------------------------------------------------------------------------
# Camera CRUD
# ---------------------------------------------------------------------------


def upsert_camera(
    session: Session,
    camera_id: str,
    stream_url: str,
    name: str | None = None,
    latitude: float = 0.0,
    longitude: float = 0.0,
    resolution: str | None = None,
    fps: int | None = None,
    status: str = "active",
    coverage_zone: dict | None = None,
) -> Camera:
    """Insert or update a camera record."""
    camera = session.get(Camera, camera_id)
    if camera is None:
        camera = Camera(
            camera_id=camera_id,
            name=name,
            latitude=latitude,
            longitude=longitude,
            stream_url=stream_url,
            resolution=resolution,
            fps=fps,
            status=status,
            coverage_zone=coverage_zone,
        )
        session.add(camera)
    else:
        camera.stream_url = stream_url
        if name is not None:
            camera.name = name
        camera.latitude = latitude
        camera.longitude = longitude
        if resolution is not None:
            camera.resolution = resolution
        if fps is not None:
            camera.fps = fps
        camera.status = status
        if coverage_zone is not None:
            camera.coverage_zone = coverage_zone
    session.flush()
    return camera


def get_camera(session: Session, camera_id: str) -> Camera | None:
    """Get a camera by ID."""
    return session.get(Camera, camera_id)


def list_cameras(session: Session, status: str | None = None) -> list[Camera]:
    """List cameras, optionally filtered by status."""
    stmt = select(Camera)
    if status is not None:
        stmt = stmt.where(Camera.status == status)
    return list(session.scalars(stmt).all())


def update_camera_status(session: Session, camera_id: str, status: str) -> None:
    """Update a camera's status."""
    camera = session.get(Camera, camera_id)
    if camera is not None:
        camera.status = status
        session.flush()


# ---------------------------------------------------------------------------
# Vehicle CRUD
# ---------------------------------------------------------------------------


def upsert_vehicle(
    session: Session,
    plate_number: str,
    vehicle_class: str | None = None,
    timestamp: datetime | None = None,
    color: str | None = None,
) -> Vehicle:
    """Find vehicle by plate or create new. Updates last_seen and increments sightings on match."""
    stmt = select(Vehicle).where(Vehicle.plate_number == plate_number)
    vehicle = session.scalars(stmt).first()
    now = timestamp or datetime.now(UTC)
    if vehicle is None:
        vehicle = Vehicle(
            plate_number=plate_number,
            vehicle_class=vehicle_class,
            color=color,
            first_seen=now,
            last_seen=now,
            total_sightings=1,
        )
        session.add(vehicle)
    else:
        vehicle.last_seen = now
        vehicle.total_sightings += 1
        if vehicle_class is not None:
            vehicle.vehicle_class = vehicle_class
        if color is not None:
            vehicle.color = color
    session.flush()
    return vehicle


def get_vehicle_by_plate(session: Session, plate_number: str) -> Vehicle | None:
    """Look up a vehicle by plate number."""
    stmt = select(Vehicle).where(Vehicle.plate_number == plate_number)
    return session.scalars(stmt).first()


def get_vehicle(session: Session, vehicle_id: uuid.UUID) -> Vehicle | None:
    """Get a vehicle by its UUID."""
    return session.get(Vehicle, vehicle_id)


# ---------------------------------------------------------------------------
# Detection CRUD
# ---------------------------------------------------------------------------


def create_detection(
    session: Session,
    camera_id: str,
    timestamp: datetime,
    frame_number: int | None = None,
    detection: DetectionResult | None = None,
    vehicle_id: uuid.UUID | None = None,
    track_id: int | None = None,
) -> Detection:
    """Create a single detection record."""
    det = Detection(
        camera_id=camera_id,
        timestamp=timestamp,
        frame_number=frame_number,
        vehicle_id=vehicle_id,
        track_id=track_id,
    )
    if detection is not None:
        det.bbox_x = detection.bbox.x
        det.bbox_y = detection.bbox.y
        det.bbox_w = detection.bbox.w
        det.bbox_h = detection.bbox.h
        det.vehicle_class = detection.vehicle_class
        det.vehicle_confidence = detection.vehicle_confidence
        det.plate_text = detection.plate_text
        det.plate_confidence = detection.plate_confidence
    session.add(det)
    session.flush()
    return det


def bulk_create_detections(
    session: Session,
    perception_result: PerceptionResult,
) -> list[Detection]:
    """Create detection records for all detections in a PerceptionResult.

    For detections with plate_text, upserts the vehicle first.
    """
    records = []
    for det in perception_result.detections:
        vehicle_id = None
        if det.plate_text:
            vehicle = upsert_vehicle(
                session,
                plate_number=det.plate_text,
                vehicle_class=det.vehicle_class,
                timestamp=perception_result.timestamp,
            )
            vehicle_id = vehicle.vehicle_id
        record = create_detection(
            session,
            camera_id=perception_result.camera_id,
            timestamp=perception_result.timestamp,
            frame_number=perception_result.frame_number,
            detection=det,
            vehicle_id=vehicle_id,
        )
        records.append(record)
    return records


def get_detections_by_camera(
    session: Session,
    camera_id: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
) -> list[Detection]:
    """Get detections for a camera within an optional time window."""
    stmt = select(Detection).where(Detection.camera_id == camera_id)
    if start_time is not None:
        stmt = stmt.where(Detection.timestamp >= start_time)
    if end_time is not None:
        stmt = stmt.where(Detection.timestamp <= end_time)
    stmt = stmt.order_by(Detection.timestamp.desc()).limit(limit)
    return list(session.scalars(stmt).all())


def get_detections_by_plate(
    session: Session,
    plate_text: str,
    limit: int = 100,
) -> list[Detection]:
    """Get detections matching a plate text string."""
    stmt = (
        select(Detection)
        .where(Detection.plate_text == plate_text)
        .order_by(Detection.timestamp.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())
