import asyncio
import sys
from loguru import logger

from project_apex.config.config import Config
from project_apex.config.environment import Environment
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.database.sqlite_market_data import SQLiteMarketDataRepository
from project_apex.api.deriv_client import DerivClient
from project_apex.data.candle_builder import CandleBuilder
from project_apex.data.tick_processor import TickCollector

async def test_engine():
    logger.info("Initializing Config and Environment...")
    config = Config()
    
    # 1. Initialize Database
    db_path = config.get_str("database", "path")
    # For testing, we can use an in-memory DB or a specific test file
    # Let's use an in-memory DB for a clean test environment
    db = SQLiteManager(":memory:")
    db.connect()
    repo = SQLiteMarketDataRepository(db)
    repo.initialize()
    logger.success("Database initialized.")

    # 2. Initialize Deriv Client
    client = DerivClient(config)
    logger.info("Connecting to Deriv API...")
    await client.connect()
    
    # 3. Initialize Candle Builder
    timeframes = config.get_list("market", "timeframes")
    async def on_candle(candle):
        logger.info(f"Built Candle: {candle}")
        
    candle_builder = CandleBuilder(timeframes=timeframes, valid_timeframes=timeframes)
    candle_builder.register_candle_callback(on_candle)
    
    # 4. Initialize Tick Collector
    symbols = config.get_list("market", "symbols")
    stats_interval = config.get_int("market", "stats_interval")
    
    collector = TickCollector(
        provider=client,
        repository=repo,
        candle_builder=candle_builder,
        symbols=symbols,
        stats_interval=stats_interval,
        valid_timeframes=timeframes
    )
    
    logger.info("Starting Tick Collector engine...")
    await collector.start()
    
    # Let it run for 10 seconds to collect data
    logger.info("Engine is running. Waiting for 10 seconds to collect ticks...")
    await asyncio.sleep(10)
    
    logger.info("Shutting down engine...")
    await client.disconnect()
    # Ensure stats are printed one last time
    logger.success("Engine test complete.")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    try:
        asyncio.run(test_engine())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
