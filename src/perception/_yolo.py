"""Shared YOLO model loading and inference base class.

Eliminates duplicated init/load/predict logic between VehicleDetector and PlateDetector.
"""

from __future__ import annotations

from loguru import logger
from ultralytics import YOLO

from src.perception._utils import resolve_device


class BaseYOLODetector:
    """Common YOLO model management for all YOLO-based detectors.

    Subclasses implement ``process()`` to convert raw YOLO results into
    domain-specific dataclasses (DetectionResult, BoundingBox, etc.).
    """

    def __init__(self, model_path: str, confidence_threshold: float, device: str) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._device = resolve_device(device)
        self._model: YOLO | None = None

    def load(self) -> None:
        """Load the YOLOv8 model weights."""
        name = type(self).__name__
        logger.info("Loading {} from {}", name, self._model_path)
        self._model = YOLO(self._model_path)
        logger.info("{} loaded on device {}", name, self._device)

    def _predict(self, source):
        """Run YOLO prediction on a source image (numpy BGR array or FrameData.image).

        Returns the raw ultralytics Results list.
        """
        return self._model.predict(
            source=source,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
