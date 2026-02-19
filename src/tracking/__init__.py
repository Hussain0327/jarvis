"""Tracking layer: single-camera MOT and cross-camera re-identification.

Imports are lazy to avoid pulling boxmot/transformers transitively
when only a subset of the package is needed.
"""

from __future__ import annotations

__all__ = [
    "CrossCameraResolver",
    "IdentityGraph",
    "ReIDEmbedder",
    "SingleCameraTracker",
]


def __getattr__(name: str):
    if name == "SingleCameraTracker":
        from src.tracking.single_camera import SingleCameraTracker

        return SingleCameraTracker
    if name == "ReIDEmbedder":
        from src.tracking.reid_embedder import ReIDEmbedder

        return ReIDEmbedder
    if name == "IdentityGraph":
        from src.tracking.identity_graph import IdentityGraph

        return IdentityGraph
    if name == "CrossCameraResolver":
        from src.tracking.cross_camera import CrossCameraResolver

        return CrossCameraResolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
