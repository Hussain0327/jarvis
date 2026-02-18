"""TrOCR license plate OCR pipeline."""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.base import BaseProcessor
from src.config import get_settings
from src.perception._utils import resolve_device


class PlateOCR(BaseProcessor[np.ndarray, tuple[str, float]]):
    """Reads license plate text from a plate crop using TrOCR."""

    def __init__(self) -> None:
        settings = get_settings().perception.plate_ocr
        self._model_name = settings.model_name
        self._confidence_threshold = settings.confidence_threshold
        self._device = resolve_device(settings.device)
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def load(self) -> None:
        """Load the TrOCR processor and model."""
        logger.info("Loading plate OCR model {}", self._model_name)
        self._processor = TrOCRProcessor.from_pretrained(self._model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name).to(self._device)
        logger.info("Plate OCR loaded on device {}", self._device)

    def process(self, input_data: np.ndarray) -> tuple[str, float]:
        """Read text from a plate crop (BGR numpy array).

        Returns (text, confidence). Returns ("", 0.0) if confidence is below threshold.
        """
        rgb = input_data[:, :, ::-1]
        pil_image = Image.fromarray(rgb)

        pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=20,
            )

        text = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

        transition_scores = self._model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True,
        )
        avg_conf = float(np.exp(transition_scores[0].cpu().numpy()).mean())

        if avg_conf < self._confidence_threshold:
            return ("", 0.0)
        return (text, avg_conf)

    def process_batch(self, inputs: list[np.ndarray]) -> list[tuple[str, float]]:
        """Run OCR on a batch of plate crops (sequential for now)."""
        return [self.process(crop) for crop in inputs]
