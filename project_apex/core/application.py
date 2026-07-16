"""
Project APEX
Application Core

This module coordinates the startup and shutdown of the application.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from project_apex.config.config import Config
from project_apex.config.environment import Environment
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.database.sqlite_market_data import SQLiteMarketDataRepository
from project_apex.api.deriv_client import DerivClient
from project_apex.api.messages import MessageBuilder
from project_apex.data.candle_builder import CandleBuilder
from project_apex.data.tick_processor import TickCollector
from project_apex.utils.logger import setup_logger


class Application:
    """Coordinates all application services."""

    def __init__(self) -> None:
        # Configure logging first
        setup_logger()

        logger.info("Starting Project APEX...")

        # Load configuration
        self.config = Config()
        self.environment = Environment()
        logger.info("Environment loaded successfully.")

        # Service references
        self.database: Optional[SQLiteManager] = None
        self.repository: Optional[SQLiteMarketDataRepository] = None
        self.deriv_client: Optional[DerivClient] = None
        self.candle_builder: Optional[CandleBuilder] = None
        self.tick_collector: Optional[TickCollector] = None

    async def initialize(self) -> None:
        """Start all required services."""
        logger.info("Initializing services...")

        # 1. Database & Repository
        db_path = self.config.get_str("database", "path")
        self.database = SQLiteManager(db_path)
        self.database.connect()

        self.repository = SQLiteMarketDataRepository(self.database)
        self.repository.initialize()

        # 2. MessageBuilder (optional, used directly in DerivClient but we can instantiate if needed)
        # 3. DerivClient
        self.deriv_client = DerivClient(self.config)

        # 4. CandleBuilder
        timeframes = self.config.get_list("market", "timeframes")
        # Ensure valid timeframes logic (for this prototype, all configured are valid)
        self.candle_builder = CandleBuilder(
            timeframes=timeframes,
            valid_timeframes=timeframes
        )

        # 5. TickCollector
        symbols = self.config.get_list("market", "symbols")
        stats_interval = self.config.get_int("market", "stats_interval")
        
        self.tick_collector = TickCollector(
            provider=self.deriv_client,
            repository=self.repository,
            candle_builder=self.candle_builder,
            symbols=symbols,
            stats_interval=stats_interval,
            valid_timeframes=timeframes
        )

        # 6. Start Client
        await self.deriv_client.connect()

        # 7. Start Collector
        await self.tick_collector.start()

        logger.success("Application initialized successfully.")

    async def shutdown(self) -> None:
        """Cleanly stop all services."""
        logger.info("Shutting down services...")

        if self.tick_collector is not None:
            await self.tick_collector.stop()

        if self.deriv_client is not None:
            await self.deriv_client.disconnect()

        if self.database is not None:
            self.database.close()

        logger.success("Application stopped successfully.")

    async def run(self) -> None:
        """Run the application until interrupted."""
        try:
            await self.initialize()
            # Keep application running
            while True:
                await asyncio.sleep(3600)
        finally:
            await self.shutdown()