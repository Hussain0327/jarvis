"""Application configuration via pydantic-settings with YAML defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def _load_yaml_config() -> dict[str, Any]:
    """Load default configuration from config.yaml."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml_defaults = _load_yaml_config()


# --- Nested config models ---


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_DATABASE__")

    host: str = "localhost"
    port: int = 5432
    name: str = "jarvis"
    user: str = "jarvis"
    password: str = "jarvis_dev"
    pool_size: int = 20
    echo: bool = False

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class VehicleDetectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_PERCEPTION_VEHICLE_DETECTOR__")

    model_path: str = "models/vehicle_detector/best.pt"
    confidence_threshold: float = 0.5
    device: str = "auto"


class PlateDetectorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_PERCEPTION_PLATE_DETECTOR__")

    model_path: str = "models/plate_detector/best.pt"
    confidence_threshold: float = 0.6
    device: str = "auto"


class PlateOCRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_PERCEPTION_PLATE_OCR__")

    model_name: str = "microsoft/trocr-base-printed"
    confidence_threshold: float = 0.7
    device: str = "auto"


class PerceptionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_PERCEPTION__")

    vehicle_detector: VehicleDetectorSettings = Field(default_factory=VehicleDetectorSettings)
    plate_detector: PlateDetectorSettings = Field(default_factory=PlateDetectorSettings)
    plate_ocr: PlateOCRSettings = Field(default_factory=PlateOCRSettings)


class TrackingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_TRACKING__")

    algorithm: str = "botsort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_INGESTION__")

    default_fps: int = 5
    frame_buffer_size: int = 100
    reconnect_max_retries: int = 10
    reconnect_backoff_base: float = 2.0


# --- Root settings ---


class Settings(BaseSettings):
    """Root application settings.

    Priority (highest wins):
    1. Environment variables (JARVIS_ prefix)
    2. .env file
    3. config/config.yaml
    4. Hardcoded defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "jarvis"
    debug: bool = False
    log_level: str = "INFO"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    @classmethod
    def from_yaml(cls) -> Settings:
        """Create settings, applying YAML config as defaults under env vars."""
        return cls(**_yaml_defaults.get("app", {}))


def get_settings() -> Settings:
    """Return application settings singleton."""
    return Settings.from_yaml()
