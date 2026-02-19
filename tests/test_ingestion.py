"""Unit tests for the ingestion layer."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.ingestion.camera_manager import CameraManager
from src.ingestion.frame_buffer import FrameBuffer
from src.ingestion.stream_handler import StreamHandler, _sanitize_url

from tests.conftest import make_frame

# ---------------------------------------------------------------------------
# URL sanitization
# ---------------------------------------------------------------------------


class TestSanitizeUrl:
    def test_strips_credentials(self):
        url = "rtsp://admin:password@192.168.1.1:554/stream"
        result = _sanitize_url(url)
        assert "admin" not in result
        assert "password" not in result
        assert "192.168.1.1" in result
        assert "554" in result

    def test_no_credentials_unchanged(self):
        url = "rtsp://192.168.1.1:554/stream"
        result = _sanitize_url(url)
        assert result == url

    def test_only_username(self):
        url = "rtsp://admin@192.168.1.1/stream"
        result = _sanitize_url(url)
        assert "admin" not in result

    def test_invalid_url_passthrough(self):
        url = "not-a-url"
        assert _sanitize_url(url) == url


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_put_get_fifo(self):
        buf = FrameBuffer("cam-001", max_size=10)
        f1 = make_frame(frame_number=1)
        f2 = make_frame(frame_number=2)
        buf.put(f1)
        buf.put(f2)
        assert buf.size == 2
        assert buf.get(timeout=0.1) is f1
        assert buf.get(timeout=0.1) is f2

    def test_drop_oldest_when_full(self):
        buf = FrameBuffer("cam-001", max_size=2)
        f1 = make_frame(frame_number=1)
        f2 = make_frame(frame_number=2)
        f3 = make_frame(frame_number=3)
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
            buf.put(make_frame(frame_number=i))
        assert buf.size == 5
        buf.clear()
        assert buf.size == 0
        assert buf.is_empty()

    def test_is_empty(self):
        buf = FrameBuffer("cam-001", max_size=10)
        assert buf.is_empty()
        buf.put(make_frame())
        assert not buf.is_empty()

    def test_dropped_count_increments(self):
        buf = FrameBuffer("cam-001", max_size=1)
        buf.put(make_frame(frame_number=1))
        buf.put(make_frame(frame_number=2))
        buf.put(make_frame(frame_number=3))
        assert buf.dropped_count == 2
        assert buf.size == 1

    def test_concurrent_put_stress(self):
        """Multiple producers should not crash or lose frames silently."""
        buf = FrameBuffer("cam-001", max_size=5)
        errors: list[Exception] = []

        def producer(n: int):
            try:
                for i in range(50):
                    buf.put(make_frame(frame_number=n * 1000 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Total puts = 200, buffer size = 5; dropped + remaining = 200
        assert buf.dropped_count + buf.size == 200


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
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        call_count = 0

        def grab_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count <= 3

        mock_cap.grab.side_effect = grab_side_effect
        mock_cap.retrieve.return_value = (True, frame)
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
        mock_cap.grab.return_value = True
        mock_cap.retrieve.return_value = (True, frame)
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
        grab_calls = [0]

        def grab_side_effect():
            grab_calls[0] += 1
            if grab_calls[0] <= 2:
                return True
            return False

        mock_cap.grab.side_effect = grab_side_effect
        mock_cap.retrieve.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
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

    @patch("src.ingestion.stream_handler.cv2.VideoCapture")
    def test_callback_exception_does_not_kill_thread(self, mock_cap_cls):
        """If the callback raises, the reader thread should continue."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.grab.return_value = True
        mock_cap.retrieve.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        call_count = 0

        def flaky_callback(frame_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Boom!")

        handler = StreamHandler(
            "cam-001", "rtsp://test", target_fps=1000,
            reconnect_max_retries=0,
        )
        handler.start(flaky_callback)
        time.sleep(0.3)
        handler.stop()

        # Should have continued after the first exception
        assert call_count >= 2


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
        mock_cap.grab.return_value = True
        mock_cap.retrieve.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
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
        frame = make_frame("cam-001")
        mgr._buffers["cam-001"].put(frame)
        result = mgr.get_frame("cam-001", timeout=0.1)
        assert result is frame

    def test_get_frame_unknown_camera(self):
        mgr = CameraManager()
        assert mgr.get_frame("nonexistent", timeout=0.1) is None

    def test_camera_id_too_long_raises(self):
        mgr = CameraManager()
        with pytest.raises(ValueError, match="1-50 chars"):
            mgr.add_camera("x" * 51, "rtsp://test")

    def test_camera_id_empty_raises(self):
        mgr = CameraManager()
        with pytest.raises(ValueError, match="1-50 chars"):
            mgr.add_camera("", "rtsp://test")

    def test_remove_camera_cleanup(self):
        """Removing a camera should clean up handler and buffer."""
        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test")
        mgr.remove_camera("cam-001")
        assert "cam-001" not in mgr._handlers
        assert "cam-001" not in mgr._buffers
        # Removing again should be a no-op
        mgr.remove_camera("cam-001")

    def test_iter_frames_with_stop_event(self):
        """iter_frames should respect stop_event."""
        mgr = CameraManager()
        mgr.add_camera("cam-001", "rtsp://test")
        # Put frames into the buffer
        for i in range(3):
            mgr._buffers["cam-001"].put(make_frame(frame_number=i))

        stop = threading.Event()
        frames = []
        for frame in mgr.iter_frames(timeout=0.05, stop_event=stop):
            frames.append(frame)
            if len(frames) >= 2:
                stop.set()
        assert len(frames) >= 2

    def test_concurrent_add_remove(self):
        """Concurrent add/remove should not crash."""
        mgr = CameraManager()
        errors: list[Exception] = []

        def adder():
            try:
                for i in range(20):
                    cid = f"cam-add-{i}"
                    try:
                        mgr.add_camera(cid, f"rtsp://test-{i}")
                    except ValueError:
                        pass  # duplicate ok
            except Exception as e:
                errors.append(e)

        def remover():
            try:
                for i in range(20):
                    mgr.remove_camera(f"cam-add-{i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=remover)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
