"""Unit tests for the tracking layer (Phase 2).

All tests mock BoxMOT, DINOv2, and database interactions.
No GPU, real models, or live database needed.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from src.base import BoundingBox

from tests.conftest import make_detection, make_frame

# ---------------------------------------------------------------------------
# Fixtures: mock boxmot module (not installed in test env)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_boxmot():
    """Inject a fake boxmot module into sys.modules for testing."""
    mock_botsort_cls = MagicMock()
    mock_bytetrack_cls = MagicMock()

    mock_module = MagicMock()
    mock_module.BotSort = mock_botsort_cls
    mock_module.ByteTrack = mock_bytetrack_cls

    old = sys.modules.get("boxmot")
    sys.modules["boxmot"] = mock_module

    # Force re-import of single_camera to pick up the mock
    sys.modules.pop("src.tracking.single_camera", None)

    yield mock_botsort_cls, mock_bytetrack_cls

    # Restore
    sys.modules.pop("src.tracking.single_camera", None)
    if old is not None:
        sys.modules["boxmot"] = old
    else:
        sys.modules.pop("boxmot", None)


# ---------------------------------------------------------------------------
# SingleCameraTracker
# ---------------------------------------------------------------------------


class TestSingleCameraTracker:
    def test_reset_creates_tracker(self, mock_boxmot):
        mock_botsort_cls, _ = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        tracker = SingleCameraTracker(device="cpu")
        mock_botsort_cls.assert_called_once()
        # Reset should re-create
        mock_botsort_cls.reset_mock()
        tracker.reset()
        mock_botsort_cls.assert_called_once()

    def test_update_empty_detections(self, mock_boxmot):
        mock_botsort_cls, _ = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        tracker = SingleCameraTracker(device="cpu")
        result = tracker.update([], make_frame())
        assert result == []
        mock_botsort_cls.return_value.update.assert_not_called()

    def test_update_returns_track_dicts(self, mock_boxmot):
        mock_botsort_cls, _ = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        # Mock tracker output: (M, 8) [x1,y1,x2,y2,track_id,conf,cls,det_index]
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.array(
            [[100, 200, 300, 400, 1, 0.92, 0, 0]], dtype=np.float32
        )
        mock_botsort_cls.return_value = mock_tracker

        tracker = SingleCameraTracker(device="cpu")
        dets = [make_detection()]
        result = tracker.update(dets, make_frame())

        assert len(result) == 1
        track = result[0]
        assert track["track_id"] == 1
        assert isinstance(track["bbox"], BoundingBox)
        assert track["bbox"].x == pytest.approx(100.0)
        assert track["bbox"].y == pytest.approx(200.0)
        assert track["bbox"].w == pytest.approx(200.0)
        assert track["bbox"].h == pytest.approx(200.0)
        assert track["confidence"] == pytest.approx(0.92)
        assert track["class_id"] == 0
        assert track["det_index"] == 0

    def test_update_attaches_detection_info(self, mock_boxmot):
        mock_botsort_cls, _ = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.array(
            [[100, 200, 300, 400, 1, 0.92, 0, 0]], dtype=np.float32
        )
        mock_botsort_cls.return_value = mock_tracker

        tracker = SingleCameraTracker(device="cpu")
        det = make_detection(plate_text="ABC1234")
        result = tracker.update([det], make_frame())

        assert result[0]["detection"] is det
        assert result[0]["detection"].plate_text == "ABC1234"

    def test_botsort_initialization(self, mock_boxmot):
        mock_botsort_cls, _ = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        SingleCameraTracker(
            algorithm="botsort",
            reid_weights="custom.pt",
            device="cpu",
            track_buffer=50,
            match_thresh=0.7,
            new_track_thresh=0.5,
            with_reid=False,
        )
        call_kwargs = mock_botsort_cls.call_args
        assert call_kwargs.kwargs["track_buffer"] == 50
        assert call_kwargs.kwargs["match_thresh"] == 0.7
        assert call_kwargs.kwargs["new_track_thresh"] == 0.5
        assert call_kwargs.kwargs["with_reid"] is False

    def test_bytetrack_fallback(self, mock_boxmot):
        _, mock_bytetrack_cls = mock_boxmot
        from src.tracking.single_camera import SingleCameraTracker

        SingleCameraTracker(algorithm="bytetrack", device="cpu")
        mock_bytetrack_cls.assert_called_once()
        call_kwargs = mock_bytetrack_cls.call_args
        assert "reid_weights" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# ReIDEmbedder
# ---------------------------------------------------------------------------


class TestReIDEmbedder:
    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_load_model(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        embedder = ReIDEmbedder(device="cpu")
        embedder.load()

        mock_proc_cls.from_pretrained.assert_called_once_with("facebook/dinov2-base")
        mock_model_cls.from_pretrained.assert_called_once_with("facebook/dinov2-base")
        mock_model.eval.assert_called_once()

    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_process_returns_768_dim(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 257, 768)
        mock_model.return_value = mock_outputs

        embedder = ReIDEmbedder(device="cpu")
        embedder.load()
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        result = embedder.process(crop)

        assert result.shape == (768,)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_process_normalizes_embedding(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        # Use a non-unit vector to verify normalization
        mock_outputs.last_hidden_state = torch.ones(1, 257, 768) * 5.0
        mock_model.return_value = mock_outputs

        embedder = ReIDEmbedder(device="cpu")
        embedder.load()
        result = embedder.process(np.zeros((100, 100, 3), dtype=np.uint8))

        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_process_bgr_to_rgb(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 257, 768)
        mock_model.return_value = mock_outputs

        embedder = ReIDEmbedder(device="cpu")
        embedder.load()

        # Create a BGR image where B=255, G=0, R=0
        bgr_crop = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr_crop[:, :, 0] = 255  # Blue channel

        embedder.process(bgr_crop)

        # The processor should receive RGB (R=0, G=0, B=255)
        call_args = mock_processor.call_args
        rgb_image = call_args.kwargs["images"]
        assert rgb_image[0, 0, 2] == 255  # What was B in BGR is now R channel pos
        assert rgb_image[0, 0, 0] == 0    # What was R in BGR is now B channel pos

    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_process_batch(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_processor.return_value = {"pixel_values": torch.zeros(3, 3, 224, 224)}

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(3, 257, 768)
        mock_model.return_value = mock_outputs

        embedder = ReIDEmbedder(device="cpu")
        embedder.load()

        crops = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = embedder.process_batch(crops)

        assert len(results) == 3
        for emb in results:
            assert emb.shape == (768,)
            assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoImageProcessor")
    def test_process_batch_empty(self, mock_proc_cls, mock_model_cls):
        from src.tracking.reid_embedder import ReIDEmbedder

        embedder = ReIDEmbedder(device="cpu")
        # No need to load for empty batch
        results = embedder.process_batch([])
        assert results == []

    def test_process_without_load_raises(self):
        from src.tracking.reid_embedder import ReIDEmbedder

        embedder = ReIDEmbedder(device="cpu")
        with pytest.raises(RuntimeError, match="load\\(\\) must be called first"):
            embedder.process(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_process_batch_without_load_raises(self):
        from src.tracking.reid_embedder import ReIDEmbedder

        embedder = ReIDEmbedder(device="cpu")
        with pytest.raises(RuntimeError, match="load\\(\\) must be called first"):
            embedder.process_batch([np.zeros((100, 100, 3), dtype=np.uint8)])


# ---------------------------------------------------------------------------
# IdentityGraph
# ---------------------------------------------------------------------------


class TestIdentityGraph:
    def test_resolve_with_plate_match(self):
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()
        vehicle_id = uuid.uuid4()

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = vehicle_id
        mock_vehicle.embedding_centroid = None
        mock_vehicle.total_sightings = 2

        with patch(
            "src.tracking.identity_graph.upsert_vehicle", return_value=mock_vehicle
        ):
            result = graph.resolve_identity(
                session=session,
                plate_text="ABC1234",
                embedding=None,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
                vehicle_class="car",
            )

        assert result == vehicle_id

    def test_resolve_with_plate_creates_new(self):
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()
        new_vehicle_id = uuid.uuid4()

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = new_vehicle_id
        mock_vehicle.embedding_centroid = None
        mock_vehicle.total_sightings = 1

        with patch(
            "src.tracking.identity_graph.upsert_vehicle", return_value=mock_vehicle
        ):
            result = graph.resolve_identity(
                session=session,
                plate_text="NEW9999",
                embedding=None,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
                vehicle_class="truck",
            )

        assert result == new_vehicle_id

    def test_resolve_with_plate_and_embedding(self):
        """When plate + embedding are both present, plate match + centroid update."""
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()
        vehicle_id = uuid.uuid4()

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = vehicle_id
        mock_vehicle.embedding_centroid = None
        mock_vehicle.total_sightings = 1

        embedding = np.random.randn(768).astype(np.float32)

        with patch(
            "src.tracking.identity_graph.upsert_vehicle", return_value=mock_vehicle
        ), patch(
            "src.tracking.identity_graph.update_vehicle_centroid"
        ) as mock_update:
            result = graph.resolve_identity(
                session=session,
                plate_text="ABC1234",
                embedding=embedding,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
            )

        assert result == vehicle_id
        mock_update.assert_called_once()

    def test_resolve_by_embedding_match(self):
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph(similarity_threshold=0.85)
        session = MagicMock()
        vehicle_id = uuid.uuid4()

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = vehicle_id
        mock_vehicle.total_sightings = 3
        mock_vehicle.embedding_centroid = np.random.randn(768).astype(
            np.float32
        ).tolist()

        embedding = np.random.randn(768).astype(np.float32)

        with patch(
            "src.tracking.identity_graph.find_similar_vehicles",
            return_value=[(mock_vehicle, 0.92)],
        ), patch("src.tracking.identity_graph.update_vehicle_centroid"):
            result = graph.resolve_identity(
                session=session,
                plate_text=None,
                embedding=embedding,
                camera_id="cam-002",
                timestamp=datetime.now(UTC),
            )

        assert result == vehicle_id

    def test_resolve_by_embedding_no_match(self):
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()

        embedding = np.random.randn(768).astype(np.float32)
        new_vehicle_id = uuid.uuid4()

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = new_vehicle_id

        with patch(
            "src.tracking.identity_graph.find_similar_vehicles", return_value=[]
        ), patch(
            "src.tracking.identity_graph.upsert_vehicle_by_embedding",
            return_value=mock_vehicle,
        ):
            result = graph.resolve_identity(
                session=session,
                plate_text=None,
                embedding=embedding,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
                vehicle_class="car",
            )

        assert result == new_vehicle_id

    def test_resolve_excludes_same_camera(self):
        """Embedding search should pass exclude_camera_id."""
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()
        embedding = np.random.randn(768).astype(np.float32)
        new_vehicle_id = uuid.uuid4()
        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = new_vehicle_id

        with patch(
            "src.tracking.identity_graph.find_similar_vehicles", return_value=[]
        ) as mock_find, patch(
            "src.tracking.identity_graph.upsert_vehicle_by_embedding",
            return_value=mock_vehicle,
        ):
            graph.resolve_identity(
                session=session,
                plate_text=None,
                embedding=embedding,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
            )

        call_kwargs = mock_find.call_args
        assert call_kwargs.kwargs["exclude_camera_id"] == "cam-001"

    def test_centroid_update(self):
        """Running average formula should produce correct result."""
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()

        # Vehicle with existing centroid seen 4 times
        old_centroid = np.array(
            [1.0, 0.0, 0.0] + [0.0] * 765, dtype=np.float32
        )
        old_centroid = old_centroid / np.linalg.norm(old_centroid)

        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = uuid.uuid4()
        mock_vehicle.total_sightings = 5  # already incremented by upsert
        mock_vehicle.embedding_centroid = old_centroid.tolist()

        new_embedding = np.array(
            [0.0, 1.0, 0.0] + [0.0] * 765, dtype=np.float32
        )
        new_embedding = new_embedding / np.linalg.norm(new_embedding)

        with patch(
            "src.tracking.identity_graph.update_vehicle_centroid"
        ) as mock_update:
            graph._update_centroid(session, mock_vehicle, new_embedding)

        mock_update.assert_called_once()
        result_centroid = mock_update.call_args[0][2]
        # Running average: (old * 4 + new) / 5, then normalize
        expected = (old_centroid * 4 + new_embedding) / 5
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_array_almost_equal(result_centroid, expected, decimal=5)

    def test_resolve_no_plate_no_embedding(self):
        """With no plate and no embedding, creates anonymous vehicle."""
        from src.tracking.identity_graph import IdentityGraph

        graph = IdentityGraph()
        session = MagicMock()
        new_vehicle_id = uuid.uuid4()
        mock_vehicle = MagicMock()
        mock_vehicle.vehicle_id = new_vehicle_id
        session.flush = MagicMock()

        with patch(
            "src.tracking.identity_graph.Vehicle", return_value=mock_vehicle,
        ):
            result = graph.resolve_identity(
                session=session,
                plate_text=None,
                embedding=None,
                camera_id="cam-001",
                timestamp=datetime.now(UTC),
                vehicle_class="car",
            )
        assert result == new_vehicle_id


# ---------------------------------------------------------------------------
# CrossCameraResolver
# ---------------------------------------------------------------------------


class TestCrossCameraResolver:
    def test_process_tracks_embeds_and_resolves(self):
        from src.tracking.cross_camera import CrossCameraResolver

        mock_embedder = MagicMock()
        mock_graph = MagicMock()

        embedding = np.random.randn(768).astype(np.float32)
        mock_embedder.process_batch.return_value = [embedding]

        vehicle_id = uuid.uuid4()
        mock_graph.resolve_identity.return_value = vehicle_id

        resolver = CrossCameraResolver(
            reid_embedder=mock_embedder, identity_graph=mock_graph
        )

        det = make_detection(plate_text="XYZ999")
        tracks = [
            {
                "track_id": 1,
                "bbox": BoundingBox(x=100, y=100, w=200, h=150),
                "confidence": 0.92,
                "class_id": 0,
                "det_index": 0,
                "detection": det,
            }
        ]

        session = MagicMock()
        frame = make_frame()
        result = resolver.process_tracks(session, tracks, frame)

        assert len(result) == 1
        assert result[0]["vehicle_id"] == vehicle_id
        assert result[0]["embedding"] is not None
        mock_graph.resolve_identity.assert_called_once()

    def test_process_tracks_empty(self):
        from src.tracking.cross_camera import CrossCameraResolver

        resolver = CrossCameraResolver(
            reid_embedder=MagicMock(), identity_graph=MagicMock()
        )
        result = resolver.process_tracks(MagicMock(), [], make_frame())
        assert result == []

    def test_process_tracks_batch_optimization(self):
        """Verify process_batch is called once, not process() N times."""
        from src.tracking.cross_camera import CrossCameraResolver

        mock_embedder = MagicMock()
        mock_graph = MagicMock()

        embeddings = [
            np.random.randn(768).astype(np.float32) for _ in range(3)
        ]
        mock_embedder.process_batch.return_value = embeddings
        mock_graph.resolve_identity.return_value = uuid.uuid4()

        resolver = CrossCameraResolver(
            reid_embedder=mock_embedder, identity_graph=mock_graph
        )

        tracks = [
            {
                "track_id": i,
                "bbox": BoundingBox(x=100, y=100, w=200, h=150),
                "confidence": 0.9,
                "class_id": 0,
                "det_index": i,
                "detection": make_detection(),
            }
            for i in range(3)
        ]

        result = resolver.process_tracks(MagicMock(), tracks, make_frame())

        assert len(result) == 3
        mock_embedder.process_batch.assert_called_once()
        mock_embedder.process.assert_not_called()
