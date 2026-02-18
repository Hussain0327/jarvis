"""Shared utilities for perception processors."""

from __future__ import annotations

import numpy as np
import torch

from src.base import BoundingBox


def resolve_device(device_str: str) -> str:
    """Resolve device string for PyTorch.

    ``'auto'`` resolves to ``'cuda:0'`` if CUDA is available, else ``'cpu'``.
    Other values (``'cpu'``, ``'cuda:0'``, etc.) are returned as-is.
    """
    if device_str == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_str


def crop_region(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop a region from an image, clamping coordinates to image bounds.

    Returns an empty array (size == 0) if the bbox is entirely outside the image.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(w, int(bbox.x + bbox.w))
    y2 = min(h, int(bbox.y + bbox.h))
    return image[y1:y2, x1:x2].copy()


def translate_bbox(inner: BoundingBox, outer: BoundingBox) -> BoundingBox:
    """Translate a bounding box from crop-local coordinates to full-frame coordinates.

    Args:
        inner: Bounding box in the crop's coordinate system.
        outer: Bounding box of the crop within the full frame.

    Returns:
        A new BoundingBox with coordinates relative to the full frame.
    """
    return BoundingBox(
        x=outer.x + inner.x,
        y=outer.y + inner.y,
        w=inner.w,
        h=inner.h,
    )
