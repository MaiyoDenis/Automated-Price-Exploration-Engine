import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from project_apex.api.deriv_client import DerivClient
from project_apex.database.sqlite_manager import SQLiteManager


class TickCollector:
    """
    Subscribes to live tick streams from the Deriv API and stores them.
    """

    def __init__(self, client: DerivClient, db: SQLiteManager, symbol: str) -> None:
        self.client = client
        self.db = db
        self.symbol = symbol
        self.is_collecting = False
        self.subscription_id: Optional[str] = None

    async def start(self) -> None:
        """Starts collecting ticks."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.client.add_message_callback(self._on_message)
        
        # Subscribe to ticks
        await self.client.send({
            "ticks": self.symbol,
            "subscribe": 1
        })
        logger.info(f"Started tick collection for {self.symbol}")

    async def stop(self) -> None:
        """Stops collecting ticks."""
        self.is_collecting = False
        if self.subscription_id and self.client.is_connected:
            await self.client.send({
                "forget": self.subscription_id
            })
            logger.info(f"Stopped tick collection for {self.symbol}")

    def _on_message(self, data: Dict[str, Any]) -> None:
        """Handles incoming tick messages."""
        if not self.is_collecting:
            return
            
        if "error" in data:
            logger.error(f"Error in tick collection: {data['error'].get('message')}")
            return
            
        if "tick" in data:
            tick_data = data["tick"]
            symbol = tick_data.get("symbol")
            
            if symbol != self.symbol:
                return
                
            epoch = tick_data.get("epoch")
            quote = tick_data.get("quote")
            self.subscription_id = tick_data.get("id")
            
            # Save to database
            self._save_tick(symbol, epoch, quote)

    def _save_tick(self, symbol: str, epoch: int, quote: float) -> None:
        """Saves a tick to the database."""
        try:
            query = "INSERT INTO ticks (symbol, timestamp, price) VALUES (?, ?, ?)"
            self.db.execute(query, (symbol, epoch, quote))
            logger.debug(f"Saved tick: {symbol} @ {quote}")
        except Exception as e:
            logger.error(f"Failed to save tick: {e}")
