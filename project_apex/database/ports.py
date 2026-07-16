"""
Project APEX
MarketDataRepository Port

Defines the abstract interface for market data storage.
"""

from abc import ABC, abstractmethod
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


class MarketDataRepository(ABC):
    """Abstract interface for storing and retrieving market data."""

    @abstractmethod
    def save_tick(self, tick: Tick) -> None:
        """Saves a tick to the repository.
        
        Args:
            tick: The Tick object to save.
        """
        pass

    @abstractmethod
    def save_candle(self, candle: Candle) -> None:
        """Saves a candle to the repository.
        
        Args:
            candle: The Candle object to save.
        """
        pass

    @abstractmethod
    def get_latest_tick_timestamp(self, symbol: str) -> int | None:
        """Retrieves the timestamp of the latest tick for a symbol.
        
        Args:
            symbol: The instrument identifier.
            
        Returns:
            The Unix epoch timestamp in milliseconds, or None if no ticks exist.
        """
        pass

    @abstractmethod
    def get_ticks(self, symbol: str, start: int, end: int) -> list[Tick]:
        """Retrieves ticks for a symbol within a time range.
        
        Args:
            symbol: The instrument identifier.
            start: Start timestamp (inclusive).
            end: End timestamp (inclusive).
            
        Returns:
            A list of Tick objects ordered by timestamp.
        """
        pass

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: int, start: int, end: int) -> list[Candle]:
        """Retrieves candles for a symbol and timeframe within a time range.
        
        Args:
            symbol: The instrument identifier.
            timeframe: The candle duration in seconds.
            start: Start timestamp (inclusive).
            end: End timestamp (inclusive).
            
        Returns:
            A list of Candle objects ordered by timestamp.
        """
        pass
