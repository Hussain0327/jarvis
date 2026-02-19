"""DINOv2/OSNet embedding extraction for vehicle re-identification (Phase 2)."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from loguru import logger

from src.base import BaseProcessor
from src.perception._utils import resolve_device


class ReIDEmbedder(BaseProcessor[np.ndarray, np.ndarray]):
    """Extracts 768-dim appearance embeddings from vehicle crops using DINOv2.

    Used for cross-camera vehicle re-identification. Crops are expected to be
    BGR numpy arrays (OpenCV convention).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device_str = device
        self._processor = None
        self._model = None
        self._device = None

    def load(self) -> None:
        """Load DINOv2 model and image processor."""
        from transformers import AutoImageProcessor, AutoModel

        self._device = torch.device(resolve_device(self._device_str))
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        logger.info(
            "Loaded ReID model {} on {}", self._model_name, self._device
        )

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Extract a 768-dim L2-normalized embedding from a single BGR crop.

        Args:
            input_data: BGR numpy array (H, W, 3).

        Returns:
            L2-normalized embedding as numpy array of shape (768,).
        """
        if self._model is None:
            raise RuntimeError("ReIDEmbedder.load() must be called first")
        # Convert BGR to RGB
        rgb = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

        # L2 normalize
        cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

        return cls_embedding.squeeze(0).cpu().numpy()

    def process_batch(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Extract embeddings for a batch of BGR crops.

        Processes crops in chunks to avoid GPU OOM on crowded frames.

        Args:
            inputs: List of BGR numpy arrays.

        Returns:
            List of L2-normalized embeddings, each shape (768,).
        """
        if not inputs:
            return []
        if self._model is None:
            raise RuntimeError("ReIDEmbedder.load() must be called first")

        max_batch_size = 32
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(inputs), max_batch_size):
            chunk = inputs[start : start + max_batch_size]
            all_embeddings.extend(self._process_chunk(chunk))
        return all_embeddings

    def _process_chunk(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Process a single chunk of crops through the model."""
        # Convert all crops BGR -> RGB
        rgb_images = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in inputs]

        processed = self._processor(images=rgb_images, return_tensors="pt")
        processed = {k: v.to(self._device) for k, v in processed.items()}

        with torch.no_grad():
            outputs = self._model(**processed)

        # CLS token for each image in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

        return [emb.cpu().numpy() for emb in cls_embeddings]
