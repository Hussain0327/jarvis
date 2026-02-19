"""Tests for base abstractions and dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import pytest
from src.base import (
    BaseAnalyzer,
    BaseProcessor,
    BaseTracker,
    BoundingBox,
    DetectionResult,
    FrameData,
    PerceptionResult,
)


class TestBoundingBox:
    def test_frozen(self):
        bbox = BoundingBox(x=1, y=2, w=3, h=4)
        with pytest.raises(FrozenInstanceError):
            bbox.x = 10  # type: ignore[misc]

    def test_values(self):
        bbox = BoundingBox(x=10.5, y=20.3, w=100.0, h=50.0)
        assert bbox.x == 10.5
        assert bbox.y == 20.3
        assert bbox.w == 100.0
        assert bbox.h == 50.0


class TestFrameData:
    def test_frozen(self):
        fd = FrameData(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            camera_id="c1",
            timestamp=datetime.now(UTC),
            frame_number=1,
        )
        with pytest.raises(FrozenInstanceError):
            fd.camera_id = "c2"  # type: ignore[misc]


class TestDetectionResult:
    def test_defaults(self):
        det = DetectionResult(
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
            vehicle_class="car",
            vehicle_confidence=0.9,
        )
        assert det.plate_bbox is None
        assert det.plate_text is None
        assert det.plate_confidence is None
        assert det.embedding is None

    def test_frozen(self):
        det = DetectionResult(
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
            vehicle_class="car",
            vehicle_confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            det.vehicle_class = "truck"  # type: ignore[misc]


class TestPerceptionResult:
    def test_default_detections(self):
        pr = PerceptionResult(
            camera_id="cam-001",
            timestamp=datetime.now(UTC),
            frame_number=1,
        )
        assert pr.detections == []

    def test_with_detections(self):
        det = DetectionResult(
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
            vehicle_class="car",
            vehicle_confidence=0.9,
        )
        pr = PerceptionResult(
            camera_id="cam-001",
            timestamp=datetime.now(UTC),
            frame_number=1,
            detections=[det],
        )
        assert len(pr.detections) == 1


class TestAbstractBases:
    def test_cannot_instantiate_base_processor(self):
        with pytest.raises(TypeError):
            BaseProcessor()  # type: ignore[abstract]

    def test_cannot_instantiate_base_tracker(self):
        with pytest.raises(TypeError):
            BaseTracker()  # type: ignore[abstract]

    def test_cannot_instantiate_base_analyzer(self):
        with pytest.raises(TypeError):
            BaseAnalyzer()  # type: ignore[abstract]
