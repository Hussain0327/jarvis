"""Tests for the configuration system."""

from __future__ import annotations

from src.config import DatabaseSettings, Settings, get_settings


def test_default_settings():
    """Settings load with hardcoded defaults when no env/yaml override."""
    settings = Settings()
    assert settings.app_name == "jarvis"
    assert settings.debug is False
    assert settings.log_level == "INFO"


def test_database_url_construction():
    """DatabaseSettings.url builds a valid PostgreSQL URL."""
    db = DatabaseSettings(host="db.test", port=5433, name="testdb", user="u", password="p")
    assert db.url == "postgresql://u:p@db.test:5433/testdb"


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("JARVIS_DEBUG", "true")
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.debug is True
    assert settings.log_level == "DEBUG"


def test_get_settings_returns_settings():
    """get_settings() returns a valid Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.database.url.startswith("postgresql://")


def test_perception_defaults():
    """Perception settings have sensible defaults."""
    settings = Settings()
    assert settings.perception.vehicle_detector.confidence_threshold == 0.5
    assert settings.perception.plate_detector.confidence_threshold == 0.6
    assert settings.perception.plate_ocr.confidence_threshold == 0.7


def test_tracking_defaults():
    """Tracking settings have sensible defaults."""
    settings = Settings()
    assert settings.tracking.algorithm == "botsort"
    assert settings.tracking.max_age == 30
