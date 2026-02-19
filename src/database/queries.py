"""CRUD query functions for Camera, Vehicle, and Detection.

All functions take an explicit Session parameter. Caller manages commit/rollback.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    if not camera_id or len(camera_id) > 50:
        raise ValueError(f"camera_id must be 1-50 chars, got {len(camera_id)!r}")
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
    """Find vehicle by plate or create new. Updates last_seen and increments sightings on match.

    Uses INSERT ... ON CONFLICT DO UPDATE to avoid race conditions under concurrent writes.
    """
    now = timestamp or datetime.now(UTC)
    values: dict = {
        "plate_number": plate_number,
        "first_seen": now,
        "last_seen": now,
        "total_sightings": 1,
    }
    if vehicle_class is not None:
        values["vehicle_class"] = vehicle_class
    if color is not None:
        values["color"] = color

    set_on_conflict: dict = {
        "last_seen": now,
        "total_sightings": Vehicle.total_sightings + 1,
    }
    if vehicle_class is not None:
        set_on_conflict["vehicle_class"] = vehicle_class
    if color is not None:
        set_on_conflict["color"] = color

    stmt = (
        pg_insert(Vehicle)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["plate_number"],
            set_=set_on_conflict,
        )
        .returning(Vehicle)
    )
    vehicle = session.execute(stmt).scalars().first()
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
    Uses a single flush at the end for better performance.
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
        record = Detection(
            camera_id=perception_result.camera_id,
            timestamp=perception_result.timestamp,
            frame_number=perception_result.frame_number,
            vehicle_id=vehicle_id,
        )
        if det is not None:
            record.bbox_x = det.bbox.x
            record.bbox_y = det.bbox.y
            record.bbox_w = det.bbox.w
            record.bbox_h = det.bbox.h
            record.vehicle_class = det.vehicle_class
            record.vehicle_confidence = det.vehicle_confidence
            record.plate_text = det.plate_text
            record.plate_confidence = det.plate_confidence
        session.add(record)
        records.append(record)
    session.flush()
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


# ---------------------------------------------------------------------------
# Embedding / Re-ID queries (Phase 2)
# ---------------------------------------------------------------------------


def find_similar_vehicles(
    session: Session,
    embedding: np.ndarray,
    threshold: float = 0.85,
    time_after: datetime | None = None,
    exclude_camera_id: str | None = None,
    limit: int = 5,
) -> list[tuple[Vehicle, float]]:
    """Find vehicles with similar embedding centroids via pgvector cosine similarity.

    Args:
        embedding: Query embedding vector (768-dim).
        threshold: Minimum cosine similarity (0-1).
        time_after: Only consider vehicles seen after this timestamp.
        exclude_camera_id: Exclude vehicles whose most recent detection
            was from this camera (for cross-camera matching).
        limit: Maximum number of results.

    Returns:
        List of (Vehicle, similarity_score) tuples ordered by similarity descending.
    """
    query_vec = embedding.tolist()
    # cosine_distance: 0 = identical, 2 = opposite
    # cosine_similarity = 1 - cosine_distance
    stmt = (
        select(
            Vehicle,
            (1 - Vehicle.embedding_centroid.cosine_distance(query_vec)).label(
                "similarity"
            ),
        )
        .filter(Vehicle.embedding_centroid.isnot(None))
        .filter(
            Vehicle.embedding_centroid.cosine_distance(query_vec) < (1 - threshold)
        )
    )

    if time_after is not None:
        stmt = stmt.filter(Vehicle.last_seen >= time_after)

    # Exclude vehicles whose latest detection is from the given camera via
    # a correlated subquery — this avoids post-hoc filtering that defeats LIMIT.
    if exclude_camera_id is not None:
        latest_camera_subq = (
            select(Detection.camera_id)
            .where(Detection.vehicle_id == Vehicle.vehicle_id)
            .order_by(Detection.timestamp.desc())
            .limit(1)
            .correlate(Vehicle)
            .scalar_subquery()
        )
        stmt = stmt.filter(latest_camera_subq != exclude_camera_id)

    stmt = (
        stmt.order_by(Vehicle.embedding_centroid.cosine_distance(query_vec))
        .limit(limit)
    )

    return [(vehicle, float(sim)) for vehicle, sim in session.execute(stmt)]


def update_vehicle_centroid(
    session: Session,
    vehicle_id: uuid.UUID,
    new_centroid: np.ndarray,
) -> None:
    """Update a vehicle's embedding centroid."""
    vehicle = session.get(Vehicle, vehicle_id)
    if vehicle is not None:
        vehicle.embedding_centroid = new_centroid.tolist()
        session.flush()


def upsert_vehicle_by_embedding(
    session: Session,
    vehicle_class: str | None,
    timestamp: datetime,
    embedding: np.ndarray,
) -> Vehicle:
    """Create a new anonymous vehicle (no plate) with an embedding centroid."""
    vehicle = Vehicle(
        vehicle_class=vehicle_class,
        first_seen=timestamp,
        last_seen=timestamp,
        total_sightings=1,
        embedding_centroid=embedding.tolist(),
    )
    session.add(vehicle)
    session.flush()
    return vehicle


def create_detection_with_track(
    session: Session,
    camera_id: str,
    timestamp: datetime,
    frame_number: int | None,
    detection: DetectionResult | None,
    vehicle_id: uuid.UUID | None,
    track_id: int | None,
    embedding: np.ndarray | None = None,
) -> Detection:
    """Create a detection record with tracking and embedding information."""
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
    if embedding is not None:
        det.embedding = embedding.tolist()
    session.add(det)
    session.flush()
    return det
