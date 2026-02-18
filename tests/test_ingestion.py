"""Unit tests for the ingestion layer."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.base import FrameData
from src.ingestion.camera_manager import CameraManager
from src.ingestion.frame_buffer import FrameBuffer
from src.ingestion.stream_handler import StreamHandler


def _make_frame(camera_id: str = "cam-001", frame_number: int = 1) -> FrameData:
    """Create a dummy FrameData for testing."""
    return FrameData(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        camera_id=camera_id,
        timestamp=datetime.now(UTC),
        frame_number=frame_number,
    )


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_put_get_fifo(self):
        buf = FrameBuffer("cam-001", max_size=10)
        f1 = _make_frame(frame_number=1)
        f2 = _make_frame(frame_number=2)
        buf.put(f1)
        buf.put(f2)
        assert buf.size == 2
        assert buf.get(timeout=0.1) is f1
        assert buf.get(timeout=0.1) is f2

    def test_drop_oldest_when_full(self):
        buf = FrameBuffer("cam-001", max_size=2)
        f1 = _make_frame(frame_number=1)
        f2 = _make_frame(frame_number=2)
        f3 = _make_frame(frame_number=3)
        buf.put(f1)
        buf.put(f2)
        buf.put(f3)
        assert buf.size == 2
        assert buf.dropped_count == 1
        result = buf.get(timeout=0.1)
        assert result.frame_number == 2

    def test_get_timeout_returns_none(self):
        buf = FrameBuffer("cam-001", max_size=10)
        assert buf.get(timeout=0.05) is None

    def test_clear(self):
        buf = FrameBuffer("cam-001", max_size=10)
        for i in range(5):
            buf.put(_make_frame(frame_number=i))
        assert buf.size == 5
        buf.clear()
        assert buf.size == 0
        assert buf.is_empty()

    def test_is_empty(self):
        buf = FrameBuffer("cam-001", max_size=10)
        assert buf.is_empty()
        buf.put(_make_frame())
        assert not buf.is_empty()

    def test_dropped_count_increments(self):
        buf = FrameBuffer("cam-001", max_size=1)
        buf.put(_make_frame(frame_number=1))
        buf.put(_make_frame(frame_number=2))
        buf.put(_make_frame(frame_number=3))
        assert buf.dropped_count == 2
        assert buf.size == 1


# ---------------------------------------------------------------------------
# StreamHandler
# ---------------------------------------------------------------------------


class TestStreamHandler:
    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_connect_success(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        handler = StreamHandler("cam-001", "rtsp://test", target_fps=5)
        assert handler._connect()
        mock_cap.set.assert_called_once()

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_connect_failure(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        handler = StreamHandler("cam-001", "rtsp://test", target_fps=5)
        assert not handler._connect()

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_start_stop_lifecycle(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # read() returns True first, then triggers stop
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        call_count = 0

        def read_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return True, frame
            return False, None

        mock_cap.read.side_effect = read_side_effect
        mock_cap_cls.return_value = mock_cap

        handler = StreamHandler(
            "cam-001", "rtsp://test", target_fps=1000,
            reconnect_max_retries=0,
        )
        callback = MagicMock()
        handler.start(callback)
        time.sleep(0.3)
        handler.stop()

        assert not handler.is_alive()
        assert callback.call_count >= 1

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_fps_throttling(self, mock_cap_cls):
        """With a very low target FPS, not every read produces a callback."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        handler = StreamHandler(
            "cam-001", "rtsp://test", target_fps=2,
            reconnect_max_retries=1,
        )
        callback = MagicMock()
        handler.start(callback)
        time.sleep(1.2)
        handler.stop()

        # At 2 FPS over ~1.2s, should get approximately 2-3 callbacks, not hundreds
        assert callback.call_count <= 5

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_reconnect_on_read_failure(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        read_calls = [0]

        def read_side_effect():
            read_calls[0] += 1
            if read_calls[0] <= 2:
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            return False, None

        mock_cap.read.side_effect = read_side_effect
        mock_cap_cls.return_value = mock_cap

        handler = StreamHandler(
            "cam-001", "rtsp://test", target_fps=1000,
            reconnect_max_retries=0, reconnect_backoff_base=0.1,
        )
        callback = MagicMock()
        handler.start(callback)
        time.sleep(0.5)
        handler.stop()

        assert callback.call_count >= 1


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------


class TestCameraManager:
    def test_add_remove_camera(self):
        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test1")
        mgr.add_camera("cam-002", "rtsp://test2")
        assert "cam-001" in mgr._handlers
        assert "cam-002" in mgr._handlers
        mgr.remove_camera("cam-001")
        assert "cam-001" not in mgr._handlers

    def test_duplicate_camera_raises(self):
        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test")
        with pytest.raises(ValueError, match="already registered"):
            mgr.add_camera("cam-001", "rtsp://test2")

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_start_stop_all(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap_cls.return_value = mock_cap

        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test1")
        mgr.add_camera("cam-002", "rtsp://test2")
        mgr.start()
        time.sleep(0.3)
        assert len(mgr.active_cameras) == 2
        mgr.stop()
        time.sleep(0.2)
        assert len(mgr.active_cameras) == 0

    def test_get_frame_from_buffer(self):
        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test")
        frame = _make_frame("cam-001")
        mgr._buffers["cam-001"].put(frame)
        result = mgr.get_frame("cam-001", timeout=0.1)
        assert result is frame

    def test_get_frame_unknown_camera(self):
        mgr = CameraManager()
        assert mgr.get_frame("nonexistent", timeout=0.1) is None
