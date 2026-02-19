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
    assert "db.test" in db.url
    assert "5433" in db.url
    assert "testdb" in db.url
    assert db.url.startswith("postgresql://")


def test_database_url_special_chars():
    """Password with special characters should be URL-encoded."""
    db = DatabaseSettings(host="h", port=5432, name="d", user="u", password="p@ss:w/rd")
    url = db.url
    # Special chars should be encoded so the URL is valid
    assert "p@ss" not in url  # @ should be encoded
    assert "postgresql://" in url


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("JARVIS_DEBUG", "true")
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.debug is True
    assert settings.log_level == "DEBUG"


def test_get_settings_returns_settings():
    """get_settings() returns a valid Settings instance."""
    get_settings.cache_clear()
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.database.url.startswith("postgresql://")
    get_settings.cache_clear()


def test_get_settings_is_singleton():
    """get_settings() should return the same object on repeated calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()


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


def test_yaml_loads_database_section():
    """from_yaml() should pick up database section from config.yaml."""
    settings = Settings.from_yaml()
    # YAML sets pool_size to 20
    assert settings.database.pool_size == 20


def test_yaml_loads_ingestion_section():
    """from_yaml() should pick up ingestion section from config.yaml."""
    settings = Settings.from_yaml()
    assert settings.ingestion.default_fps == 5
    assert settings.ingestion.frame_buffer_size == 100


def test_yaml_loads_tracking_section():
    """from_yaml() should pick up tracking section from config.yaml."""
    settings = Settings.from_yaml()
    assert settings.tracking.iou_threshold == 0.3
