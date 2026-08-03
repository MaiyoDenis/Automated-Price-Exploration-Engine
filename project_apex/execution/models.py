"""
Project APEX — Execution Domain Models

Represents open positions and completed (closed) trades.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from project_apex.risk.models import Direction


@dataclass
class Position:
    """
    An open (active) trade position.

    Attributes:
        id: Unique position identifier (UUID).
        symbol: Trading instrument.
        direction: LONG or SHORT.
        size: Position size (stake / units).
        entry_price: Fill price at open.
        stop_loss: Price level at which position is auto-closed as a loss.
        take_profit: Price level at which position is auto-closed as a profit.
        opened_at: Unix epoch milliseconds when the position was opened.
        strategy_name: Strategy that generated the order.
        metadata: Passthrough metadata from the order.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: Direction = Direction.LONG
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    opened_at: int = 0  # epoch ms
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Compute unrealized P&L at ``current_price``.

        For LONG: (current_price - entry_price) * size
        For SHORT: (entry_price - current_price) * size
        """
        if self.direction == Direction.LONG:
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L as fraction of the initial stake (entry_price * size)."""
        stake = self.entry_price * self.size
        if stake == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / stake

    def is_stop_hit(self, current_price: float) -> bool:
        """True if current_price has breached the stop-loss level."""
        if self.stop_loss <= 0:
            return False
        if self.direction == Direction.LONG:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_tp_hit(self, current_price: float) -> bool:
        """True if current_price has reached the take-profit level."""
        if self.take_profit <= 0:
            return False
        if self.direction == Direction.LONG:
            return current_price >= self.take_profit
        return current_price <= self.take_profit


@dataclass(frozen=True)
class Trade:
    """
    A completed (closed) trade record.

    Attributes:
        id: Position ID that was closed.
        symbol: Trading instrument.
        direction: LONG or SHORT.
        size: Position size.
        entry_price: Price at open.
        exit_price: Price at close.
        opened_at: Epoch ms when opened.
        closed_at: Epoch ms when closed.
        realized_pnl: Absolute profit/loss.
        realized_pnl_pct: P&L as fraction of entry stake.
        close_reason: Why the position was closed (e.g. "stop_loss", "take_profit", "manual").
        strategy_name: Originating strategy.
    """

    id: str
    symbol: str
    direction: Direction
    size: float
    entry_price: float
    exit_price: float
    opened_at: int
    closed_at: int
    realized_pnl: float
    realized_pnl_pct: float
    close_reason: str
    strategy_name: str
