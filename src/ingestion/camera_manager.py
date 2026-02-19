"""Multi-camera orchestration."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from loguru import logger

from src.base import FrameData
from src.config import get_settings
from src.ingestion.frame_buffer import FrameBuffer
from src.ingestion.stream_handler import StreamHandler


class CameraManager:
    """Manages multiple camera streams with per-camera buffers."""

    def __init__(self) -> None:
        self._settings = get_settings().ingestion
        self._handlers: dict[str, StreamHandler] = {}
        self._buffers: dict[str, FrameBuffer] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        stream_url: str,
        target_fps: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        """Register a new camera stream."""
        if len(camera_id) < 1 or len(camera_id) > 50:
            raise ValueError(f"camera_id must be 1-50 chars, got {len(camera_id)}")
        with self._lock:
            if camera_id in self._handlers:
                raise ValueError(f"Camera {camera_id!r} already registered")
            self._handlers[camera_id] = StreamHandler(
                camera_id=camera_id,
                stream_url=stream_url,
                target_fps=target_fps,
            )
            self._buffers[camera_id] = FrameBuffer(
                camera_id=camera_id,
                max_size=buffer_size,
            )
        logger.info("Added camera {}", camera_id)

    def remove_camera(self, camera_id: str) -> None:
        """Stop and remove a camera."""
        with self._lock:
            handler = self._handlers.pop(camera_id, None)
            buf = self._buffers.pop(camera_id, None)
        if handler is not None:
            handler.stop()
        if buf is not None:
            buf.clear()
        logger.info("Removed camera {}", camera_id)

    def start(self) -> None:
        """Start all registered camera streams."""
        with self._lock:
            items = list(self._handlers.items())
        for camera_id, handler in items:
            with self._lock:
                buf = self._buffers.get(camera_id)
            if buf is not None:
                handler.start(frame_callback=buf.put)
        logger.info("Started {} camera streams", len(items))

    def stop(self) -> None:
        """Stop all camera streams."""
        with self._lock:
            handlers = list(self._handlers.values())
        for handler in handlers:
            handler.stop()
        logger.info("Stopped all camera streams")

    def get_frame(self, camera_id: str, timeout: float = 1.0) -> FrameData | None:
        """Pull a frame from a specific camera buffer."""
        with self._lock:
            buf = self._buffers.get(camera_id)
        if buf is None:
            return None
        return buf.get(timeout=timeout)

    def iter_frames(
        self, timeout: float = 0.1, stop_event: threading.Event | None = None,
    ) -> Iterator[FrameData]:
        """Round-robin yield frames from all camera buffers.

        Continues indefinitely until stop_event is set (or no cameras remain).
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            with self._lock:
                camera_ids = list(self._buffers.keys())
            if not camera_ids:
                return
            for camera_id in camera_ids:
                with self._lock:
                    buf = self._buffers.get(camera_id)
                if buf is None:
                    continue
                frame = buf.get(timeout=timeout)
                if frame is not None:
                    yield frame

    @property
    def active_cameras(self) -> list[str]:
        """IDs of cameras with running streams."""
        with self._lock:
            return [cid for cid, h in self._handlers.items() if h.is_alive()]
