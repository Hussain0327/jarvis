"""Ingestion layer: camera feed connection and frame extraction.

Imports are lazy to avoid pulling cv2 transitively when only a subset
of the package is needed (e.g. ``FrameBuffer`` does not require OpenCV).
"""

from __future__ import annotations

__all__ = ["CameraManager", "FrameBuffer", "StreamHandler"]


def __getattr__(name: str):
    if name == "CameraManager":
        from src.ingestion.camera_manager import CameraManager

        return CameraManager
    if name == "FrameBuffer":
        from src.ingestion.frame_buffer import FrameBuffer

        return FrameBuffer
    if name == "StreamHandler":
        from src.ingestion.stream_handler import StreamHandler

        return StreamHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
