"""
Project APEX
MarketDataProvider Port

Defines the abstract interface for broker clients.
"""

from abc import ABC, abstractmethod
from project_apex.models.tick import Tick


class MarketDataProvider(ABC):
    """Abstract interface for all broker market data clients."""

    @abstractmethod
    async def connect(self) -> None:
        """Establishes connection and authenticates with the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnects from the broker."""
        pass

    @abstractmethod
    async def subscribe_ticks(self, symbol: str) -> str:
        """Subscribes to live ticks for a symbol.
        
        Args:
            symbol: The instrument identifier.
            
        Returns:
            The subscription identifier to be used for unsubscription.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribes from a live tick stream.
        
        Args:
            subscription_id: The ID returned by subscribe_ticks.
        """
        pass

    @abstractmethod
    async def receive(self) -> Tick:
        """Receives the next tick from the provider.
        
        Returns:
            The next Tick object.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Indicates whether the provider is connected and authenticated."""
        pass
