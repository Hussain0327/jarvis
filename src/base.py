"""Base abstractions and shared dataclasses for all pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, TypeVar

import numpy as np

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box (x, y, w, h) in pixel coordinates."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class FrameData:
    """A single timestamped frame from a camera."""

    image: np.ndarray
    camera_id: str
    timestamp: datetime
    frame_number: int


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """One detected vehicle in a single frame."""

    bbox: BoundingBox
    vehicle_class: str
    vehicle_confidence: float
    plate_bbox: BoundingBox | None = None
    plate_text: str | None = None
    plate_confidence: float | None = None
    embedding: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """All detections for a single frame."""

    camera_id: str
    timestamp: datetime
    frame_number: int
    detections: list[DetectionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class BaseProcessor(ABC, Generic[InputT, OutputT]):
    """Stateless inference processor (e.g. YOLO, TrOCR).

    Subclasses load a model once, then process individual inputs.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and prepare for inference."""

    @abstractmethod
    def process(self, input_data: InputT) -> OutputT:
        """Run inference on a single input."""

    @abstractmethod
    def process_batch(self, inputs: list[InputT]) -> list[OutputT]:
        """Run inference on a batch of inputs."""


class BaseTracker(ABC):
    """Stateful multi-object tracker (e.g. BoT-SORT, cross-camera).

    Maintains internal state across frames.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""

    @abstractmethod
    def update(self, detections: list[DetectionResult], frame: FrameData) -> list[dict]:
        """Update tracks with new detections from a frame.

        Returns a list of track dicts with at least 'track_id' and 'bbox'.
        """


class BaseAnalyzer(ABC, Generic[InputT, OutputT]):
    """Analytics processor (e.g. clustering, anomaly detection).

    May be stateful (fitted model) or stateless (rule-based).
    """

    @abstractmethod
    def fit(self, data: InputT) -> None:
        """Fit/train the analyzer on historical data."""

    @abstractmethod
    def analyze(self, data: InputT) -> OutputT:
        """Run analysis on new data and return results."""
