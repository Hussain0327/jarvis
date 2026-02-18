"""Multi-camera orchestration."""

from __future__ import annotations

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

    def add_camera(
        self,
        camera_id: str,
        stream_url: str,
        target_fps: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        """Register a new camera stream."""
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
        handler = self._handlers.pop(camera_id, None)
        if handler is not None:
            handler.stop()
        buf = self._buffers.pop(camera_id, None)
        if buf is not None:
            buf.clear()
        logger.info("Removed camera {}", camera_id)

    def start(self) -> None:
        """Start all registered camera streams."""
        for camera_id, handler in self._handlers.items():
            buf = self._buffers[camera_id]
            handler.start(frame_callback=buf.put)
        logger.info("Started {} camera streams", len(self._handlers))

    def stop(self) -> None:
        """Stop all camera streams."""
        for handler in self._handlers.values():
            handler.stop()
        logger.info("Stopped all camera streams")

    def get_frame(self, camera_id: str, timeout: float = 1.0) -> FrameData | None:
        """Pull a frame from a specific camera buffer."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return None
        return buf.get(timeout=timeout)

    def iter_frames(self, timeout: float = 0.1) -> Iterator[FrameData]:
        """Round-robin yield frames from all camera buffers."""
        camera_ids = list(self._buffers.keys())
        while camera_ids:
            yielded = False
            for camera_id in camera_ids:
                frame = self._buffers[camera_id].get(timeout=timeout)
                if frame is not None:
                    yielded = True
                    yield frame
            if not yielded:
                break

    @property
    def active_cameras(self) -> list[str]:
        """IDs of cameras with running streams."""
        return [cid for cid, h in self._handlers.items() if h.is_alive()]
