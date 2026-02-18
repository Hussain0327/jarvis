"""Perception layer: vehicle detection, plate detection, plate OCR.

Imports are lazy to avoid pulling torch/ultralytics/transformers transitively
when only a subset of the package is needed.
"""

from __future__ import annotations

__all__ = ["PerceptionPipeline", "PlateDetector", "PlateOCR", "VehicleDetector"]


def __getattr__(name: str):
    if name == "PerceptionPipeline":
        from src.perception.perception_pipeline import PerceptionPipeline

        return PerceptionPipeline
    if name == "PlateDetector":
        from src.perception.plate_detector import PlateDetector

        return PlateDetector
    if name == "PlateOCR":
        from src.perception.plate_ocr import PlateOCR

        return PlateOCR
    if name == "VehicleDetector":
        from src.perception.vehicle_detector import VehicleDetector

        return VehicleDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
