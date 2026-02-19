"""Cross-camera identity resolution via plate matching and embedding similarity (Phase 2)."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from src.base import FrameData
from src.perception._utils import crop_region
from src.tracking.identity_graph import IdentityGraph
from src.tracking.reid_embedder import ReIDEmbedder


class CrossCameraResolver:
    """Coordinates ReIDEmbedder and IdentityGraph to resolve tracked detections
    from any camera to persistent vehicle identities.
    """

    def __init__(
        self,
        reid_embedder: ReIDEmbedder | None = None,
        identity_graph: IdentityGraph | None = None,
    ) -> None:
        self._reid_embedder = reid_embedder or ReIDEmbedder()
        self._identity_graph = identity_graph or IdentityGraph()

    def load(self) -> None:
        """Load the ReID embedding model."""
        self._reid_embedder.load()

    def process_tracks(
        self,
        session: Session,
        tracks: list[dict],
        frame: FrameData,
    ) -> list[dict]:
        """For each track, extract an embedding and resolve vehicle identity.

        Args:
            session: SQLAlchemy session (caller manages commit/rollback).
            tracks: Track dicts from SingleCameraTracker.update().
            frame: The original FrameData for this frame.

        Returns:
            Enriched track dicts with 'vehicle_id' and 'embedding' added.
        """
        if not tracks:
            return []

        # Collect crops for all tracks with valid bounding boxes
        crops: list[np.ndarray] = []
        crop_indices: list[int] = []

        for i, track in enumerate(tracks):
            bbox = track["bbox"]
            crop = crop_region(frame.image, bbox)
            if crop.size > 0:
                crops.append(crop)
                crop_indices.append(i)

        # Batch embed all crops at once
        embeddings: list[np.ndarray] = []
        if crops:
            embeddings = self._reid_embedder.process_batch(crops)

        # Map embeddings back to tracks
        embedding_map: dict[int, np.ndarray] = {}
        for idx, emb in zip(crop_indices, embeddings):
            embedding_map[idx] = emb

        # Resolve identities
        results = []
        for i, track in enumerate(tracks):
            enriched = dict(track)
            embedding = embedding_map.get(i)

            # Extract plate text from the attached detection
            plate_text = None
            detection = track.get("detection")
            if detection is not None:
                plate_text = detection.plate_text

            vehicle_class = None
            if detection is not None:
                vehicle_class = detection.vehicle_class

            vehicle_id = self._identity_graph.resolve_identity(
                session=session,
                plate_text=plate_text,
                embedding=embedding,
                camera_id=frame.camera_id,
                timestamp=frame.timestamp,
                vehicle_class=vehicle_class,
            )

            enriched["vehicle_id"] = vehicle_id
            enriched["embedding"] = embedding
            results.append(enriched)

        logger.debug(
            "Resolved {} tracks to vehicle identities for camera {}",
            len(results),
            frame.camera_id,
        )
        return results
