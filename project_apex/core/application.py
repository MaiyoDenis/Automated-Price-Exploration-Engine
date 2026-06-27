"""
Project APEX
Application Core

This module coordinates the startup and shutdown of the application.
"""

from __future__ import annotations

from loguru import logger

from project_apex.config.config import Config
from project_apex.database.sqlite_manager import Database
from project_apex.utils.logger import setup_logger


class Application:
    """Coordinates all application services."""

    def __init__(self) -> None:
        # Configure logging first
        setup_logger()

        logger.info("Starting Project APEX...")

        # Load configuration
        self.config = Config()

        # Initialize database manager
        self.database = Database(
            self.config.get("database", "path")
        )

    def initialize(self) -> None:
        """Start all required services."""

        logger.info("Initializing services...")

        self.database.connect()
        self.database.initialize()

        logger.success("Application initialized successfully.")

    def shutdown(self) -> None:
        """Cleanly stop all services."""

        logger.info("Shutting down services...")

        self.database.close()

        logger.success("Application stopped successfully.")