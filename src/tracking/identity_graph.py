"""Vehicle identity management and cross-camera graph (Phase 2)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from src.database.models import Vehicle
from src.database.queries import (
    find_similar_vehicles,
    update_vehicle_centroid,
    upsert_vehicle,
    upsert_vehicle_by_embedding,
)


class IdentityGraph:
    """Manages the merging of vehicle identities from multiple cameras.

    Decides whether two sightings are the same vehicle using plate matching
    (authoritative) or embedding similarity (fallback). Maintains per-vehicle
    centroid embeddings as a running average.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        time_window_seconds: float = 3600.0,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._time_window_seconds = time_window_seconds

    def resolve_identity(
        self,
        session: Session,
        plate_text: str | None,
        embedding: np.ndarray | None,
        camera_id: str,
        timestamp: datetime,
        vehicle_class: str | None = None,
    ) -> uuid.UUID:
        """Resolve a sighting to a persistent vehicle identity.

        Priority:
        1. Plate match (authoritative)
        2. Embedding similarity (cross-camera only)
        3. Create anonymous vehicle

        Returns:
            The vehicle_id for this sighting.
        """
        if plate_text is not None:
            vehicle = self._match_by_plate(
                session, plate_text, timestamp, vehicle_class
            )
            if embedding is not None:
                self._update_centroid(session, vehicle, embedding)
            return vehicle.vehicle_id

        if embedding is not None:
            vehicle = self._match_by_embedding(
                session, embedding, camera_id, timestamp
            )
            if vehicle is not None:
                vehicle.last_seen = timestamp
                vehicle.total_sightings += 1
                self._update_centroid(session, vehicle, embedding)
                session.flush()
                return vehicle.vehicle_id

        # No plate, no embedding match -> create anonymous vehicle
        vehicle = self._create_anonymous_vehicle(
            session, embedding, vehicle_class, timestamp
        )
        return vehicle.vehicle_id

    def _match_by_plate(
        self,
        session: Session,
        plate_text: str,
        timestamp: datetime,
        vehicle_class: str | None,
    ) -> Vehicle:
        """Match by exact plate number, upserting if necessary."""
        vehicle = upsert_vehicle(
            session,
            plate_number=plate_text,
            vehicle_class=vehicle_class,
            timestamp=timestamp,
        )
        logger.debug("Plate match: {} -> {}", plate_text, vehicle.vehicle_id)
        return vehicle

    def _match_by_embedding(
        self,
        session: Session,
        embedding: np.ndarray,
        camera_id: str,
        timestamp: datetime,
    ) -> Vehicle | None:
        """Find a similar vehicle via pgvector cosine similarity search.

        Excludes vehicles whose latest detection is from the same camera.
        Only considers vehicles seen within the configured time window.
        """
        time_after = timestamp - timedelta(seconds=self._time_window_seconds)

        matches = find_similar_vehicles(
            session,
            embedding=embedding,
            threshold=self._similarity_threshold,
            time_after=time_after,
            exclude_camera_id=camera_id,
            limit=1,
        )

        if matches:
            vehicle, similarity = matches[0]
            logger.debug(
                "Embedding match: vehicle={} sim={:.3f}",
                vehicle.vehicle_id,
                similarity,
            )
            return vehicle

        return None

    def _create_anonymous_vehicle(
        self,
        session: Session,
        embedding: np.ndarray | None,
        vehicle_class: str | None,
        timestamp: datetime,
    ) -> Vehicle:
        """Create a new vehicle without a plate number."""
        if embedding is not None:
            vehicle = upsert_vehicle_by_embedding(
                session,
                vehicle_class=vehicle_class,
                timestamp=timestamp,
                embedding=embedding,
            )
        else:
            vehicle = Vehicle(
                vehicle_class=vehicle_class,
                first_seen=timestamp,
                last_seen=timestamp,
                total_sightings=1,
            )
            session.add(vehicle)
            session.flush()

        logger.debug("Created anonymous vehicle: {}", vehicle.vehicle_id)
        return vehicle

    def _update_centroid(
        self,
        session: Session,
        vehicle: Vehicle,
        new_embedding: np.ndarray,
    ) -> None:
        """Update the vehicle's embedding centroid using running average."""
        if vehicle.embedding_centroid is None:
            update_vehicle_centroid(
                session, vehicle.vehicle_id, new_embedding
            )
            return

        n = vehicle.total_sightings
        old_centroid = np.array(vehicle.embedding_centroid, dtype=np.float32)
        new_centroid = (old_centroid * (n - 1) + new_embedding) / n
        # Re-normalize to unit length
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        update_vehicle_centroid(session, vehicle.vehicle_id, new_centroid)
