import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from project_apex.api.deriv_client import DerivClient
from project_apex.database.sqlite_manager import SQLiteManager


class CandleDownloader:
    """
    Downloads historical OHLCV data from the Deriv API.
    """

    def __init__(self, client: DerivClient, db: SQLiteManager) -> None:
        self.client = client
        self.db = db
        
        # We need a way to correlate responses to requests, 
        # so we keep track of pending requests by req_id
        self.client.add_message_callback(self._on_message)
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self._next_req_id = 1

    async def download(self, symbol: str, granularity: int = 60, count: int = 1000) -> None:
        """
        Downloads history for a given symbol.
        granularity: duration of each candle in seconds
        count: number of candles
        """
        req_id = self._next_req_id
        self._next_req_id += 1
        
        request = {
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "candles",
            "granularity": granularity,
            "req_id": req_id
        }
        
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = future
        
        logger.info(f"Downloading {count} candles for {symbol} (granularity: {granularity})")
        await self.client.send(request)
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=30.0)
            self._save_candles(symbol, granularity, response)
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for history data for {symbol}")
            if req_id in self.pending_requests:
                del self.pending_requests[req_id]

    def _on_message(self, data: Dict[str, Any]) -> None:
        """Handles incoming messages to resolve pending futures."""
        req_id = data.get("req_id")
        if req_id and req_id in self.pending_requests:
            future = self.pending_requests.pop(req_id)
            if not future.done():
                if "error" in data:
                    future.set_exception(Exception(data["error"].get("message", "Unknown error")))
                else:
                    future.set_result(data)

    def _save_candles(self, symbol: str, granularity: int, data: Dict[str, Any]) -> None:
        """Saves downloaded candles to the database."""
        if "candles" not in data:
            logger.warning(f"No candles found in response: {data}")
            return
            
        candles = data["candles"]
        logger.info(f"Saving {len(candles)} candles to database for {symbol}")
        
        # Batch insert
        try:
            query = """
            INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            values = [
                (
                    symbol,
                    granularity,
                    c.get("epoch"),
                    c.get("open"),
                    c.get("high"),
                    c.get("low"),
                    c.get("close")
                )
                for c in candles
            ]
            
            self.db.execute_many(query, values)
            logger.success(f"Successfully saved candles for {symbol}")
        except Exception as e:
            logger.error(f"Failed to save candles: {e}")
