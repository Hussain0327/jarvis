"""Unit tests for the perception layer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from src.base import BoundingBox, DetectionResult, FrameData, PerceptionResult
from src.perception._utils import crop_region, resolve_device, translate_bbox
from src.perception.perception_pipeline import PerceptionPipeline, to_boxmot_format


def _make_frame(camera_id: str = "cam-001", frame_number: int = 1) -> FrameData:
    return FrameData(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_id=camera_id,
        timestamp=datetime.now(UTC),
        frame_number=frame_number,
    )


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert resolve_device("cuda:0") == "cuda:0"

    def test_auto_resolves(self):
        result = resolve_device("auto")
        assert result in ("cpu", "cuda:0")


# ---------------------------------------------------------------------------
# VehicleDetector
# ---------------------------------------------------------------------------


class TestVehicleDetector:
    @patch("src.perception._yolo.YOLO")
    def test_load(self, mock_yolo_cls):
        from src.perception.vehicle_detector import VehicleDetector

        det = VehicleDetector()
        det.load()
        mock_yolo_cls.assert_called_once_with(det._model_path)

    @patch("src.perception._yolo.YOLO")
    def test_process_returns_detections(self, mock_yolo_cls):
        from src.perception.vehicle_detector import VehicleDetector

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock YOLO results
        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
        mock_boxes.conf = torch.tensor([0.92])
        mock_boxes.cls = torch.tensor([0.0])
        mock_boxes.__len__ = lambda self: 1
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {0: "car"}
        mock_model.predict.return_value = [mock_result]

        det = VehicleDetector()
        det.load()
        frame = _make_frame()
        results = det.process(frame)

        assert len(results) == 1
        assert results[0].vehicle_class == "car"
        assert results[0].vehicle_confidence == pytest.approx(0.92)
        assert results[0].bbox.x == pytest.approx(100.0)
        assert results[0].bbox.y == pytest.approx(200.0)
        assert results[0].bbox.w == pytest.approx(200.0)
        assert results[0].bbox.h == pytest.approx(200.0)

    @patch("src.perception._yolo.YOLO")
    def test_process_empty_detections(self, mock_yolo_cls):
        from src.perception.vehicle_detector import VehicleDetector

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([]).reshape(0, 4)
        mock_boxes.conf = torch.tensor([])
        mock_boxes.cls = torch.tensor([])
        mock_boxes.__len__ = lambda self: 0
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.names = {}
        mock_model.predict.return_value = [mock_result]

        det = VehicleDetector()
        det.load()
        results = det.process(_make_frame())
        assert results == []


# ---------------------------------------------------------------------------
# PlateDetector
# ---------------------------------------------------------------------------


class TestPlateDetector:
    @patch("src.perception._yolo.YOLO")
    def test_process_returns_bboxes(self, mock_yolo_cls):
        from src.perception.plate_detector import PlateDetector

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 110.0, 50.0]])
        mock_boxes.conf = torch.tensor([0.85])
        mock_boxes.__len__ = lambda self: 1
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        det = PlateDetector()
        det.load()
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        plates = det.process(crop)

        assert len(plates) == 1
        assert plates[0].x == pytest.approx(10.0)
        assert plates[0].w == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# PlateOCR
# ---------------------------------------------------------------------------


class TestPlateOCR:
    @patch("src.perception.plate_ocr.VisionEncoderDecoderModel")
    @patch("src.perception.plate_ocr.TrOCRProcessor")
    def test_process_returns_text_and_confidence(self, mock_proc_cls, mock_model_cls):
        from src.perception.plate_ocr import PlateOCR

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value.pixel_values = torch.zeros(1, 3, 384, 384)
        mock_processor.batch_decode.return_value = ["ABC1234"]

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3]])
        mock_outputs.scores = (torch.tensor([[0.0]]),)
        mock_model.generate.return_value = mock_outputs

        # transition_scores with high confidence
        transition_scores = torch.tensor([[-0.05, -0.03, -0.02]])
        mock_model.compute_transition_scores.return_value = transition_scores

        ocr = PlateOCR()
        ocr.load()
        crop = np.zeros((50, 100, 3), dtype=np.uint8)
        text, conf = ocr.process(crop)

        assert text == "ABC1234"
        assert conf > 0.9

    @patch("src.perception.plate_ocr.VisionEncoderDecoderModel")
    @patch("src.perception.plate_ocr.TrOCRProcessor")
    def test_low_confidence_returns_empty(self, mock_proc_cls, mock_model_cls):
        from src.perception.plate_ocr import PlateOCR

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value.pixel_values = torch.zeros(1, 3, 384, 384)
        mock_processor.batch_decode.return_value = ["???"]

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3]])
        mock_outputs.scores = (torch.tensor([[0.0]]),)
        mock_model.generate.return_value = mock_outputs

        # Very low confidence
        transition_scores = torch.tensor([[-5.0, -5.0, -5.0]])
        mock_model.compute_transition_scores.return_value = transition_scores

        ocr = PlateOCR()
        ocr.load()
        text, conf = ocr.process(np.zeros((50, 100, 3), dtype=np.uint8))

        assert text == ""
        assert conf == 0.0


# ---------------------------------------------------------------------------
# Pipeline helpers (real numpy, no mocks)
# ---------------------------------------------------------------------------


class TestCropRegion:
    def test_basic_crop(self):
        image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        bbox = BoundingBox(x=100, y=50, w=200, h=100)
        crop = crop_region(image, bbox)
        assert crop.shape == (100, 200, 3)

    def test_clamps_to_image_bounds(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=-10, y=-10, w=50, h=50)
        crop = crop_region(image, bbox)
        assert crop.shape == (40, 40, 3)

    def test_out_of_bounds_returns_empty(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=200, y=200, w=50, h=50)
        crop = crop_region(image, bbox)
        assert crop.size == 0


class TestTranslateBbox:
    def test_basic_translation(self):
        inner = BoundingBox(x=10, y=20, w=50, h=30)
        outer = BoundingBox(x=100, y=200, w=300, h=400)
        result = translate_bbox(inner, outer)
        assert result.x == 110
        assert result.y == 220
        assert result.w == 50
        assert result.h == 30


class TestToBoxmotFormat:
    def test_basic_conversion(self):
        dets = [
            DetectionResult(
                bbox=BoundingBox(x=10, y=20, w=100, h=50),
                vehicle_class="car",
                vehicle_confidence=0.9,
            ),
            DetectionResult(
                bbox=BoundingBox(x=200, y=300, w=80, h=60),
                vehicle_class="truck",
                vehicle_confidence=0.85,
            ),
        ]
        result = to_boxmot_format(dets)
        assert result.shape == (2, 6)
        assert result.dtype == np.float32
        # First row: x1=10, y1=20, x2=110, y2=70, conf=0.9, cls=0
        np.testing.assert_array_almost_equal(result[0], [10, 20, 110, 70, 0.9, 0])
        # Second row: x1=200, y1=300, x2=280, y2=360, conf=0.85, cls=1
        np.testing.assert_array_almost_equal(result[1], [200, 300, 280, 360, 0.85, 1])

    def test_empty_detections(self):
        result = to_boxmot_format([])
        assert result.shape == (0, 6)


# ---------------------------------------------------------------------------
# PerceptionPipeline (mocked sub-processors)
# ---------------------------------------------------------------------------


class TestPerceptionPipeline:
    def test_full_pipeline(self):
        """Integration test with mocked sub-processors."""
        mock_vehicle_det = MagicMock()
        mock_plate_det = MagicMock()
        mock_ocr = MagicMock()

        vehicle_detection = DetectionResult(
            bbox=BoundingBox(x=100, y=100, w=200, h=150),
            vehicle_class="car",
            vehicle_confidence=0.95,
        )
        mock_vehicle_det.process.return_value = [vehicle_detection]
        mock_plate_det.process.return_value = [BoundingBox(x=10, y=80, w=80, h=25)]
        mock_ocr.process.return_value = ("ABC1234", 0.92)

        pipeline = PerceptionPipeline(
            vehicle_detector=mock_vehicle_det,
            plate_detector=mock_plate_det,
            plate_ocr=mock_ocr,
        )

        frame = FrameData(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_id="cam-001",
            timestamp=datetime.now(UTC),
            frame_number=1,
        )
        result = pipeline.process(frame)

        assert isinstance(result, PerceptionResult)
        assert result.camera_id == "cam-001"
        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.plate_text == "ABC1234"
        assert det.plate_confidence == pytest.approx(0.92)
        # plate bbox should be translated to full-frame coords
        assert det.plate_bbox.x == pytest.approx(110)
        assert det.plate_bbox.y == pytest.approx(180)

    def test_graceful_degradation_on_ocr_failure(self):
        """If OCR fails, the detection should still be kept with plate bbox."""
        mock_vehicle_det = MagicMock()
        mock_plate_det = MagicMock()
        mock_ocr = MagicMock()

        vehicle_detection = DetectionResult(
            bbox=BoundingBox(x=100, y=100, w=200, h=150),
            vehicle_class="car",
            vehicle_confidence=0.95,
        )
        mock_vehicle_det.process.return_value = [vehicle_detection]
        mock_plate_det.process.return_value = [BoundingBox(x=10, y=80, w=80, h=25)]
        mock_ocr.process.side_effect = RuntimeError("OCR failed")

        pipeline = PerceptionPipeline(
            vehicle_detector=mock_vehicle_det,
            plate_detector=mock_plate_det,
            plate_ocr=mock_ocr,
        )

        frame = FrameData(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_id="cam-001",
            timestamp=datetime.now(UTC),
            frame_number=1,
        )
        result = pipeline.process(frame)

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.plate_bbox is not None
        assert det.plate_text is None

    def test_no_plates_found(self):
        """If no plates are found, detection is kept without plate info."""
        mock_vehicle_det = MagicMock()
        mock_plate_det = MagicMock()
        mock_ocr = MagicMock()

        vehicle_detection = DetectionResult(
            bbox=BoundingBox(x=100, y=100, w=200, h=150),
            vehicle_class="car",
            vehicle_confidence=0.95,
        )
        mock_vehicle_det.process.return_value = [vehicle_detection]
        mock_plate_det.process.return_value = []

        pipeline = PerceptionPipeline(
            vehicle_detector=mock_vehicle_det,
            plate_detector=mock_plate_det,
            plate_ocr=mock_ocr,
        )

        frame = FrameData(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_id="cam-001",
            timestamp=datetime.now(UTC),
            frame_number=1,
        )
        result = pipeline.process(frame)

        assert len(result.detections) == 1
        assert result.detections[0].plate_text is None
        mock_ocr.process.assert_not_called()
