"""Logging setup using loguru: human-readable console + JSON file output."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = "logs/jarvis.log") -> None:
    """Configure loguru with console and optional file sinks.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path for JSON log file. None to disable file logging.
    """
    logger.remove()

    # Console: human-readable with color
    console_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=console_fmt)

    # File: JSON for machine parsing
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format="{message}",
            serialize=True,
            rotation="50 MB",
            retention="7 days",
        )
