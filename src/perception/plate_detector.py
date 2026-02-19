"""YOLOv8 license plate detection wrapper."""

from __future__ import annotations

import numpy as np

from src.base import BaseProcessor, BoundingBox
from src.config import get_settings
from src.perception._yolo import BaseYOLODetector


class PlateDetector(BaseYOLODetector, BaseProcessor[np.ndarray, list[BoundingBox]]):
    """Detects license plates within a vehicle crop using YOLOv8.

    Input is a BGR numpy array (vehicle crop). Output is plate bounding boxes
    in crop-local coordinates.
    """

    def __init__(self) -> None:
        settings = get_settings().perception.plate_detector
        super().__init__(settings.model_path, settings.confidence_threshold, settings.device)

    def process(self, input_data: np.ndarray) -> list[BoundingBox]:
        """Run plate detection on a vehicle crop (BGR numpy array)."""
        results = self._predict(input_data)
        boxes = results[0].boxes

        plates = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            plates.append(BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        return plates

    def process_batch(self, inputs: list[np.ndarray]) -> list[list[BoundingBox]]:
        """Run detection on a batch of crops via true batch inference."""
        if not inputs:
            return []
        if self._model is None:
            raise RuntimeError("PlateDetector.load() must be called first")
        batch_results = self._model.predict(
            source=inputs,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
        all_plates = []
        for result in batch_results:
            boxes = result.boxes
            plates = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                plates.append(BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
            all_plates.append(plates)
        return all_plates
