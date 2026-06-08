"""Logging configuration using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path | str] = None,
    serialize: bool = False,
) -> None:
    """Configure Loguru logging for the pipeline.

    Args:
        log_level: Minimum log level to display.
        log_file: Optional path to a log file.
        serialize: If True, output JSON-formatted logs.
    """
    logger.remove()  # Remove default handler

    # Console handler with Rich-compatible formatting
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        serialize=serialize,
    )

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            rotation="10 MB",
            retention="7 days",
            serialize=serialize,
        )

    logger.debug(f"Logging configured (level={log_level}, file={log_file})")


def get_logger():
    """Return the configured Loguru logger instance."""
    return logger
