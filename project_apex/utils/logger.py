"""
Project APEX
Logging System
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger() -> None:
    """Configure application logging."""

    # Remove the default logger
    logger.remove()

    # Ensure the logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Console output
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )

    # File output
    logger.add(
        "logs/apex.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        enqueue=True,
    )