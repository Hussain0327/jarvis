"""End-to-end frame processing: detection -> plate -> OCR."""

from __future__ import annotations

import numpy as np
from loguru import logger

from src.base import (
    BaseProcessor,
    BoundingBox,
    DetectionResult,
    FrameData,
    PerceptionResult,
)
from src.perception._utils import crop_region, translate_bbox
from src.perception.plate_detector import PlateDetector
from src.perception.plate_ocr import PlateOCR
from src.perception.vehicle_detector import VehicleDetector


class PerceptionPipeline(BaseProcessor[FrameData, PerceptionResult]):
    """Chains vehicle detection -> plate detection -> plate OCR.

    Each sub-processor can be injected via the constructor (useful for testing).
    Degrades gracefully: if plate detection or OCR fails on one vehicle, the
    detection is kept with whatever information was gathered.
    """

    def __init__(
        self,
        vehicle_detector: VehicleDetector | None = None,
        plate_detector: PlateDetector | None = None,
        plate_ocr: PlateOCR | None = None,
    ) -> None:
        self._vehicle_detector = vehicle_detector or VehicleDetector()
        self._plate_detector = plate_detector or PlateDetector()
        self._plate_ocr = plate_ocr or PlateOCR()

    def load(self) -> None:
        """Load all sub-processor models."""
        self._vehicle_detector.load()
        self._plate_detector.load()
        self._plate_ocr.load()

    def process(self, input_data: FrameData) -> PerceptionResult:
        """Run the full perception pipeline on a single frame."""
        image = input_data.image
        vehicle_detections = self._vehicle_detector.process(input_data)
        enriched = [self._enrich_detection(det, image) for det in vehicle_detections]

        return PerceptionResult(
            camera_id=input_data.camera_id,
            timestamp=input_data.timestamp,
            frame_number=input_data.frame_number,
            detections=enriched,
        )

    def process_batch(self, inputs: list[FrameData]) -> list[PerceptionResult]:
        """Run the pipeline on a batch of frames (sequential for now)."""
        return [self.process(frame) for frame in inputs]

    def _enrich_detection(
        self, det: DetectionResult, image: np.ndarray
    ) -> DetectionResult:
        """Attempt to find and read a license plate for a single vehicle detection.

        Returns the original detection enriched with plate info, or unchanged
        if plate detection/OCR fails or produces no results.
        """
        vehicle_crop = crop_region(image, det.bbox)
        if vehicle_crop.size == 0:
            return det

        plate_bbox_local = self._detect_plate(vehicle_crop)
        if plate_bbox_local is None:
            return det

        plate_bbox_full = translate_bbox(plate_bbox_local, det.bbox)
        plate_text, plate_conf = self._read_plate(vehicle_crop, plate_bbox_local)

        return DetectionResult(
            bbox=det.bbox,
            vehicle_class=det.vehicle_class,
            vehicle_confidence=det.vehicle_confidence,
            plate_bbox=plate_bbox_full,
            plate_text=plate_text if plate_text else None,
            plate_confidence=plate_conf if plate_conf > 0.0 else None,
        )

    def _detect_plate(self, vehicle_crop: np.ndarray) -> BoundingBox | None:
        """Run plate detection on a vehicle crop, returning the first plate found."""
        try:
            plate_bboxes = self._plate_detector.process(vehicle_crop)
        except Exception:
            logger.opt(exception=True).debug("Plate detection failed for a vehicle crop")
            return None
        return plate_bboxes[0] if plate_bboxes else None

    def _read_plate(
        self, vehicle_crop: np.ndarray, plate_bbox: BoundingBox
    ) -> tuple[str, float]:
        """Crop the plate region and run OCR. Returns ("", 0.0) on failure."""
        plate_crop = crop_region(vehicle_crop, plate_bbox)
        if plate_crop.size == 0:
            return ("", 0.0)
        try:
            return self._plate_ocr.process(plate_crop)
        except Exception:
            logger.opt(exception=True).debug("Plate OCR failed for a plate crop")
            return ("", 0.0)


VEHICLE_CLASS_IDS: dict[str, int] = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "van": 4,
    "bicycle": 5,
}
"""Stable class-name-to-integer mapping for BoxMOT.

Unknown classes get ``len(VEHICLE_CLASS_IDS)`` as a fallback.
"""

_UNKNOWN_CLASS_ID = len(VEHICLE_CLASS_IDS)


def to_boxmot_format(detections: list[DetectionResult]) -> np.ndarray:
    """Convert detections to BoxMOT format: ``(N, 6) [x1, y1, x2, y2, conf, cls]``.

    Class names are mapped to stable integer IDs via :data:`VEHICLE_CLASS_IDS`.
    Returns an empty ``(0, 6)`` array if *detections* is empty.
    """
    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    rows = []
    for det in detections:
        cls_id = VEHICLE_CLASS_IDS.get(det.vehicle_class, _UNKNOWN_CLASS_ID)
        rows.append([
            det.bbox.x,
            det.bbox.y,
            det.bbox.x + det.bbox.w,
            det.bbox.y + det.bbox.h,
            det.vehicle_confidence,
            cls_id,
        ])

    return np.array(rows, dtype=np.float32)
