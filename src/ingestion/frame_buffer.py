"""Thread-safe bounded frame buffer with drop-oldest policy."""

from __future__ import annotations

import queue
import threading

from src.base import FrameData
from src.config import get_settings


class FrameBuffer:
    """Thread-safe bounded buffer for camera frames.

    Uses a drop-oldest policy when full to ensure real-time freshness.
    The ``put`` method is safe to call from a reader thread while ``get``
    and ``dropped_count`` are read from the consumer thread.
    """

    def __init__(self, camera_id: str, max_size: int | None = None) -> None:
        if max_size is None:
            max_size = get_settings().ingestion.frame_buffer_size
        self._camera_id = camera_id
        self._queue: queue.Queue[FrameData] = queue.Queue(maxsize=max_size)
        self._dropped_count = 0
        self._drop_lock = threading.Lock()

    @property
    def camera_id(self) -> str:
        """Camera ID this buffer belongs to."""
        return self._camera_id

    def put(self, frame: FrameData) -> None:
        """Add a frame, dropping the oldest if the buffer is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
                with self._drop_lock:
                    self._dropped_count += 1
            except queue.Empty:
                pass
        self._queue.put_nowait(frame)

    def get(self, timeout: float = 1.0) -> FrameData | None:
        """Get the next frame, returning ``None`` on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Drain all frames from the buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def size(self) -> int:
        """Current number of frames in the buffer."""
        return self._queue.qsize()

    @property
    def dropped_count(self) -> int:
        """Total number of frames dropped due to full buffer (thread-safe)."""
        with self._drop_lock:
            return self._dropped_count

    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._queue.empty()
