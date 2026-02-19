"""Application configuration via pydantic-settings with YAML defaults."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

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
    password: str = ""
    pool_size: int = 20
    echo: bool = False

    @property
    def url(self) -> str:
        return URL.create(
            drivername="postgresql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.name,
        ).render_as_string(hide_password=False)


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
    reid_weights: str = "osnet_x0_25_msmt17.pt"
    track_buffer: int = 30
    match_thresh: float = 0.8
    new_track_thresh: float = 0.6
    with_reid: bool = True


class ReIDSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_REID__")

    model_name: str = "facebook/dinov2-base"
    device: str = "auto"
    similarity_threshold: float = 0.85
    time_window_seconds: float = 3600.0


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
    reid: ReIDSettings = Field(default_factory=ReIDSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    @classmethod
    def from_yaml(cls) -> Settings:
        """Create settings, applying YAML config as defaults under env vars."""
        yaml_data = dict(_yaml_defaults)
        # Merge top-level app settings
        init: dict[str, Any] = {}
        if "app" in yaml_data:
            init.update(yaml_data["app"])
        # Map each sub-section to corresponding nested settings
        for key in ("database", "perception", "tracking", "reid", "ingestion"):
            if key in yaml_data:
                init[key] = yaml_data[key]
        return cls(**init)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return application settings singleton."""
    return Settings.from_yaml()
