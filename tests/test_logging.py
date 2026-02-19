"""Tests for logging setup."""

from __future__ import annotations

import json

from loguru import logger
from src.logging import setup_logging


def test_setup_logging_console_only():
    """setup_logging with log_file=None should not raise."""
    setup_logging(level="DEBUG", log_file=None)
    # Should have at least the console handler
    logger.info("test message")


def test_setup_logging_creates_log_dir(tmp_path):
    """setup_logging should create parent directories for the log file."""
    nested_log = str(tmp_path / "deep" / "nested" / "app.log")
    setup_logging(level="INFO", log_file=nested_log)
    logger.info("test message")
    # Verify the directory was created
    assert (tmp_path / "deep" / "nested").is_dir()


def test_log_file_is_json(tmp_path):
    """The file sink should produce JSON-serialized log lines."""
    log_file = str(tmp_path / "test.log")
    setup_logging(level="INFO", log_file=log_file)
    logger.info("hello from test")
    # Force flush by removing the sink
    logger.remove()

    with open(log_file) as f:
        content = f.read().strip()
    assert content  # not empty
    # Each line should be valid JSON
    for line in content.splitlines():
        data = json.loads(line)
        assert "text" in data


def test_setup_logging_removes_previous_handlers():
    """Calling setup_logging twice should not duplicate handlers."""
    setup_logging(level="INFO", log_file=None)
    handler_count_1 = len(logger._core.handlers)
    setup_logging(level="DEBUG", log_file=None)
    handler_count_2 = len(logger._core.handlers)
    # logger.remove() is called first, so count should stay the same
    assert handler_count_2 == handler_count_1


def test_setup_logging_respects_level(tmp_path):
    """Messages below the configured level should not appear in the file."""
    log_file = str(tmp_path / "level.log")
    setup_logging(level="WARNING", log_file=log_file)
    logger.debug("should not appear")
    logger.info("should not appear")
    logger.warning("should appear")
    logger.remove()

    with open(log_file) as f:
        content = f.read()
    assert "should appear" in content
    assert "should not appear" not in content
