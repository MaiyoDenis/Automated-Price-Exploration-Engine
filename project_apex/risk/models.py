"""
Project APEX — Risk Engine Models

Defines the data structures that flow between the StrategyEngine
and the RiskEngine, and then to the ExecutionEngine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Direction(Enum):
    """Trade direction."""
    LONG = auto()
    SHORT = auto()


@dataclass(frozen=True)
class TradeOrder:
    """
    An order approved by the RiskEngine to be sent to the ExecutionEngine.

    Attributes:
        symbol: Trading instrument.
        direction: LONG or SHORT.
        size: Fractional or unit position size (e.g. stake in USD).
        entry_price: Expected fill price (market).
        stop_loss: Absolute price level for stop-loss (0.0 = no stop).
        take_profit: Absolute price level for take-profit (0.0 = no TP).
        strategy_name: Originating strategy name.
        signal_confidence: Confidence of the original signal (0-1).
        metadata: Passthrough metadata from signal.
    """
    symbol: str
    direction: Direction
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    signal_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    """
    The RiskEngine's verdict on a TradeSignal.

    Attributes:
        approved: Whether the trade may proceed.
        reason: Human-readable explanation (especially for rejections).
        order: The (possibly size-adjusted) TradeOrder, or ``None`` if rejected.
    """
    approved: bool
    reason: str
    order: TradeOrder | None = None
