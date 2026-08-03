"""
Project APEX — Paper Broker

Simulates trade execution with realistic slippage and spread,
without making any real network calls.

Security note: No credentials, tokens, or external connections are used.
All computation is purely in-memory.
"""
from __future__ import annotations

import time
from typing import Callable, Awaitable

from loguru import logger

from project_apex.risk.models import TradeOrder, Direction
from project_apex.execution.models import Position, Trade


TradeHandler = Callable[[Trade], Awaitable[None]]


class PaperBroker:
    """
    Simulated order execution engine.

    Applies configurable slippage and spread to simulate real market execution.
    Maintains open positions in memory and emits :class:`Trade` records on close.

    Args:
        slippage_pct: Fraction of price applied as slippage per fill (e.g., 0.0001 = 0.01%).
        spread_pct: Bid-ask spread as fraction of price (applied on entry).
    """

    def __init__(
        self,
        slippage_pct: float = 0.0001,
        spread_pct: float = 0.0002,
    ) -> None:
        self._slippage_pct = slippage_pct
        self._spread_pct = spread_pct
        self._open_positions: dict[str, Position] = {}  # position_id → Position
        self._trade_handlers: list[TradeHandler] = []

        logger.info(
            f"[PaperBroker] Initialized | "
            f"slippage={slippage_pct:.4%} spread={spread_pct:.4%}"
        )

    def add_trade_handler(self, handler: TradeHandler) -> None:
        """Register an async handler called when a trade is closed."""
        self._trade_handlers.append(handler)

    # ── Position management ───────────────────────────────────────────────────

    async def open_position(self, order: TradeOrder, current_time_ms: int | None = None) -> Position:
        """
        Open a new position from an approved TradeOrder.

        Applies spread and slippage to simulate realistic fill price.
        """
        fill_price = self._simulate_fill(order.entry_price, order.direction, side="open")
        opened_at = current_time_ms or int(time.time() * 1000)

        position = Position(
            symbol=order.symbol,
            direction=order.direction,
            size=order.size,
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=opened_at,
            strategy_name=order.strategy_name,
            metadata=order.metadata,
        )
        self._open_positions[position.id] = position

        logger.info(
            f"[PaperBroker] OPENED {order.direction.name} {order.symbol} | "
            f"size={order.size:.4f} fill={fill_price:.5f} "
            f"SL={order.stop_loss:.5f} TP={order.take_profit:.5f} "
            f"id={position.id[:8]}"
        )
        return position

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
        current_time_ms: int | None = None,
    ) -> Trade | None:
        """
        Close an open position at ``exit_price``.

        Returns the completed :class:`Trade` or ``None`` if position not found.
        """
        position = self._open_positions.pop(position_id, None)
        if position is None:
            logger.warning(f"[PaperBroker] Position {position_id[:8]} not found for close.")
            return None

        fill_price = self._simulate_fill(exit_price, position.direction, side="close")
        closed_at = current_time_ms or int(time.time() * 1000)

        if position.direction == Direction.LONG:
            realized_pnl = (fill_price - position.entry_price) * position.size
        else:
            realized_pnl = (position.entry_price - fill_price) * position.size

        stake = position.entry_price * position.size
        realized_pnl_pct = realized_pnl / stake if stake != 0 else 0.0

        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=fill_price,
            opened_at=position.opened_at,
            closed_at=closed_at,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            close_reason=reason,
            strategy_name=position.strategy_name,
        )

        pnl_str = f"+{realized_pnl:.4f}" if realized_pnl >= 0 else f"{realized_pnl:.4f}"
        logger.info(
            f"[PaperBroker] CLOSED {position.direction.name} {position.symbol} | "
            f"exit={fill_price:.5f} PnL={pnl_str} ({realized_pnl_pct:.1%}) "
            f"reason={reason} id={position.id[:8]}"
        )

        for handler in self._trade_handlers:
            await handler(trade)

        return trade

    async def check_stops_and_targets(
        self,
        symbol: str,
        current_price: float,
        current_time_ms: int | None = None,
    ) -> list[Trade]:
        """
        Check all open positions for a symbol against stop-loss and take-profit.
        Called by the tick pipeline on each new price.
        Returns a list of any trades that were automatically closed.
        """
        closed_trades: list[Trade] = []
        for pos_id, pos in list(self._open_positions.items()):
            if pos.symbol != symbol:
                continue

            if pos.is_stop_hit(current_price):
                trade = await self.close_position(
                    pos_id, current_price, reason="stop_loss", current_time_ms=current_time_ms
                )
                if trade:
                    closed_trades.append(trade)

            elif pos.is_tp_hit(current_price):
                trade = await self.close_position(
                    pos_id, current_price, reason="take_profit", current_time_ms=current_time_ms
                )
                if trade:
                    closed_trades.append(trade)

        return closed_trades

    @property
    def open_positions(self) -> dict[str, Position]:
        """Read-only view of open positions."""
        return dict(self._open_positions)

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    # ── Simulation helpers ────────────────────────────────────────────────────

    def _simulate_fill(self, price: float, direction: Direction, side: str) -> float:
        """
        Applies spread and slippage to a nominal price.

        On open (entry):
          - LONG:  spread increases cost (buy at ask).
          - SHORT: spread decreases revenue (sell at bid).
        On close (exit):
          - LONG:  we sell at bid (price - spread).
          - SHORT: we buy at ask (price + spread).
        Slippage is always adverse (increases cost or reduces proceeds).
        """
        spread = price * self._spread_pct / 2.0
        slippage = price * self._slippage_pct

        if side == "open":
            if direction == Direction.LONG:
                return price + spread + slippage  # Buy at ask + slippage
            else:
                return price - spread - slippage  # Sell at bid - slippage
        else:  # close
            if direction == Direction.LONG:
                return price - spread - slippage  # Exit long: sell at bid - slippage
            else:
                return price + spread + slippage  # Exit short: buy at ask + slippage
