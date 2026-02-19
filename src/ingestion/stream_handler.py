"""Per-camera RTSP/MJPEG reader thread."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlparse, urlunparse

import cv2
from loguru import logger

from src.base import FrameData
from src.config import get_settings


def _sanitize_url(url: str) -> str:
    """Strip credentials from a URL for safe logging."""
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            # Replace netloc user:pass with ***
            safe_netloc = "***:***@" + parsed.hostname
            if parsed.port:
                safe_netloc += f":{parsed.port}"
            return urlunparse(parsed._replace(netloc=safe_netloc))
    except Exception:
        pass
    return url

# OpenCV capture configuration
_OPEN_TIMEOUT_MS = 5000
_READ_TIMEOUT_MS = 5000
_CAP_BUFFER_SIZE = 1

# Thread management
_THREAD_JOIN_TIMEOUT_S = 5.0
_MAX_BACKOFF_DELAY_S = 60.0


class StreamHandler:
    """Reads frames from a single camera stream in a dedicated background thread.

    OpenCV ``VideoCapture`` is not thread-safe, so each camera gets its own
    daemon thread that reads frames and emits them via a callback.
    """

    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        target_fps: int | None = None,
        reconnect_max_retries: int | None = None,
        reconnect_backoff_base: float | None = None,
    ) -> None:
        settings = get_settings().ingestion
        self._camera_id = camera_id
        self._stream_url = stream_url
        self._target_fps = target_fps or settings.default_fps
        self._reconnect_max_retries = reconnect_max_retries or settings.reconnect_max_retries
        self._reconnect_backoff_base = reconnect_backoff_base or settings.reconnect_backoff_base

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_number = 0

    @property
    def camera_id(self) -> str:
        """Camera ID this handler reads from."""
        return self._camera_id

    def start(self, frame_callback: Callable[[FrameData], None]) -> None:
        """Spawn the reader thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("StreamHandler {} already running", self._camera_id)
            return
        self._stop_event.clear()
        self._frame_number = 0
        self._thread = threading.Thread(
            target=self._reader_loop,
            args=(frame_callback,),
            daemon=True,
            name=f"stream-{self._camera_id}",
        )
        self._thread.start()
        logger.info("Started stream reader for camera {}", self._camera_id)

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            self._thread = None
        self._disconnect()
        logger.info("Stopped stream reader for camera {}", self._camera_id)

    def is_alive(self) -> bool:
        """Check if the reader thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Open the video capture with timeout protection."""
        self._disconnect()
        self._cap = cv2.VideoCapture(
            self._stream_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MS,
            ],
        )
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, _CAP_BUFFER_SIZE)
            logger.info("Connected to stream {}", _sanitize_url(self._stream_url))
            return True
        logger.warning("Failed to connect to stream {}", _sanitize_url(self._stream_url))
        self._disconnect()
        return False

    def _disconnect(self) -> None:
        """Release the video capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reconnect_with_backoff(self) -> bool:
        """Attempt reconnection with exponential backoff (capped at 60 s)."""
        for attempt in range(1, self._reconnect_max_retries + 1):
            if self._stop_event.is_set():
                return False
            delay = min(self._reconnect_backoff_base ** attempt, _MAX_BACKOFF_DELAY_S)
            logger.info(
                "Reconnect attempt {}/{} for {} in {:.1f}s",
                attempt, self._reconnect_max_retries, self._camera_id, delay,
            )
            self._stop_event.wait(delay)
            if self._stop_event.is_set():
                return False
            if self._connect():
                return True
        logger.error(
            "Exhausted {} reconnect attempts for {}",
            self._reconnect_max_retries, self._camera_id,
        )
        return False

    # ------------------------------------------------------------------
    # Reader loop
    # ------------------------------------------------------------------

    def _reader_loop(self, frame_callback: Callable[[FrameData], None]) -> None:
        """Main loop: read frames, throttle to target FPS, emit via callback."""
        if not self._connect():
            if not self._reconnect_with_backoff():
                return

        min_interval = 1.0 / self._target_fps
        last_emit_time = 0.0

        while not self._stop_event.is_set():
            if not self._cap.grab():
                logger.warning("Read failure on camera {}", self._camera_id)
                if not self._reconnect_with_backoff():
                    break
                continue

            now = time.monotonic()
            if now - last_emit_time < min_interval:
                continue

            ret, frame = self._cap.retrieve()
            if not ret:
                continue

            last_emit_time = now
            self._frame_number += 1
            frame_data = FrameData(
                image=frame,
                camera_id=self._camera_id,
                timestamp=datetime.now(UTC),
                frame_number=self._frame_number,
            )
            try:
                frame_callback(frame_data)
            except Exception:
                logger.opt(exception=True).error(
                    "Frame callback error on camera {}", self._camera_id
                )

        self._disconnect()
