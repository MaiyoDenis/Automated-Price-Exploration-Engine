import asyncio
import sys
from loguru import logger
from project_apex.config.config import Config
from project_apex.api.deriv_client import DerivClient

async def test_connection():
    config = Config()
    client = DerivClient(config)
    
    logger.info("Connecting to Deriv WebSocket...")
    await client.connect()
    
    if client.is_connected:
        logger.success("Connected and authorized!")
    
    # Subscribe to a test symbol
    symbol = "R_25"
    logger.info(f"Subscribing to {symbol}...")
    await client.subscribe_ticks(symbol)
    
    # Wait for 3 ticks
    ticks_received = 0
    while ticks_received < 3:
        tick = await client.receive()
        logger.info(f"Received tick: {tick}")
        ticks_received += 1
        
    logger.info("Unsubscribing and disconnecting...")
    await client.unsubscribe(symbol)
    await client.disconnect()
    logger.success("Test completed successfully.")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception("Test failed")
