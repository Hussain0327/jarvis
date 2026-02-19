"""BoT-SORT/ByteTrack per-camera multi-object tracking (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from src.base import BaseTracker, BoundingBox, DetectionResult, FrameData
from src.perception._utils import resolve_device
from src.perception.perception_pipeline import to_boxmot_format


class SingleCameraTracker(BaseTracker):
    """Wraps BoxMOT's BoT-SORT (or ByteTrack fallback) for single-camera MOT.

    One instance should be created per camera. The tracker is stateful and must
    receive frames in order.
    """

    def __init__(
        self,
        algorithm: str = "botsort",
        reid_weights: str = "osnet_x0_25_msmt17.pt",
        device: str = "auto",
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        new_track_thresh: float = 0.6,
        with_reid: bool = True,
    ) -> None:
        self._algorithm = algorithm
        self._reid_weights = reid_weights
        self._device_str = device
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._new_track_thresh = new_track_thresh
        self._with_reid = with_reid
        self._tracker = None
        self.reset()

    def reset(self) -> None:
        """Re-create the underlying BoxMOT tracker."""
        device = torch.device(resolve_device(self._device_str))

        if self._algorithm == "bytetrack":
            from boxmot import ByteTrack

            self._tracker = ByteTrack(
                track_buffer=self._track_buffer,
                match_thresh=self._match_thresh,
            )
            logger.info("Initialized ByteTrack (motion-only)")
        else:
            from boxmot import BotSort

            self._tracker = BotSort(
                reid_weights=Path(self._reid_weights),
                device=device,
                half=False,
                track_buffer=self._track_buffer,
                match_thresh=self._match_thresh,
                new_track_thresh=self._new_track_thresh,
                with_reid=self._with_reid,
            )
            logger.info("Initialized BoT-SORT (appearance + motion)")

    def update(
        self, detections: list[DetectionResult], frame: FrameData
    ) -> list[dict]:
        """Update tracks with new detections from a frame.

        Returns a list of track dicts with keys: track_id, bbox, confidence,
        class_id, det_index, and optionally detection (the original DetectionResult).
        """
        dets_array = to_boxmot_format(detections)

        if dets_array.shape[0] == 0:
            return []

        # BoxMOT returns (M, 8): [x1, y1, x2, y2, track_id, conf, cls, det_index]
        tracks_array = self._tracker.update(dets_array, frame.image)

        if tracks_array.shape[0] == 0:
            return []

        results = []
        for row in tracks_array:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            det_index = int(row[7])

            track_dict = {
                "track_id": int(row[4]),
                "bbox": BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                "confidence": float(row[5]),
                "class_id": int(row[6]),
                "det_index": det_index,
            }

            # Attach original detection if index is valid
            if 0 <= det_index < len(detections):
                track_dict["detection"] = detections[det_index]

            results.append(track_dict)

        return results
