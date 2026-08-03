"""
Project APEX — Trade Signal Models

Defines the canonical signal types and the TradeSignal dataclass that flows
from Strategy → RiskEngine → ExecutionEngine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SignalType(Enum):
    """All possible signal types a strategy can emit."""

    BUY = auto()          # Open a long position
    SELL = auto()         # Open a short position
    CLOSE_LONG = auto()   # Close an existing long position
    CLOSE_SHORT = auto()  # Close an existing short position
    HOLD = auto()         # Do nothing


@dataclass(frozen=True)
class TradeSignal:
    """
    An immutable signal emitted by a strategy.

    Attributes:
        symbol: The trading instrument (e.g. ``"R_25"``).
        signal_type: Buy, sell, hold, or close.
        confidence: Strategy confidence score in [0.0, 1.0]. Higher = stronger signal.
        price: The price at which the signal was generated.
        timestamp: Unix epoch milliseconds of the triggering candle/tick.
        strategy_name: Name of the strategy that produced this signal.
        metadata: Optional dict with extra context (indicator values, reasons, etc.).
    """

    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: int
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")
        if self.timestamp <= 0:
            raise ValueError(f"timestamp must be positive, got {self.timestamp}")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
