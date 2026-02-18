"""YOLOv8 vehicle detection wrapper."""

from __future__ import annotations

from src.base import BaseProcessor, BoundingBox, DetectionResult, FrameData
from src.config import get_settings
from src.perception._yolo import BaseYOLODetector


class VehicleDetector(BaseYOLODetector, BaseProcessor[FrameData, list[DetectionResult]]):
    """Detects vehicles in a frame using YOLOv8.

    Converts YOLO xyxy format to BoundingBox(x, y, w, h) where (x, y) is top-left.
    """

    def __init__(self) -> None:
        settings = get_settings().perception.vehicle_detector
        super().__init__(settings.model_path, settings.confidence_threshold, settings.device)

    def process(self, input_data: FrameData) -> list[DetectionResult]:
        """Run vehicle detection on a single frame."""
        results = self._predict(input_data.image)
        boxes = results[0].boxes
        names = results[0].names

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append(
                DetectionResult(
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    vehicle_class=names[int(boxes.cls[i])],
                    vehicle_confidence=float(boxes.conf[i]),
                )
            )
        return detections

    def process_batch(self, inputs: list[FrameData]) -> list[list[DetectionResult]]:
        """Run detection on a batch of frames (sequential for now)."""
        return [self.process(frame) for frame in inputs]
