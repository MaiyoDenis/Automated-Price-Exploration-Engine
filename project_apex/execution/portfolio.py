"""
Project APEX — Portfolio Tracker

Tracks equity, open positions, realized P&L, and daily performance metrics.
Serves as the source of truth for the RiskEngine's portfolio state getters.
All calculations use double-entry bookkeeping to prevent drift.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from loguru import logger

from project_apex.execution.models import Position, Trade
from project_apex.risk.models import Direction


class Portfolio:
    """
    Real-time portfolio state tracker.

    Maintains:
    - Cash balance (starts at ``initial_capital``).
    - Open positions (by position ID).
    - Closed trade history.
    - Daily P&L reset at midnight (UTC).
    - Peak equity for drawdown calculation.

    The RiskEngine uses :meth:`daily_pnl_pct`, :meth:`drawdown_pct`,
    :meth:`open_position_count`, and :meth:`equity` as live state getters.

    Args:
        initial_capital: Starting cash in account currency (e.g. USD).
    """

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._open_positions: dict[str, Position] = {}
        self._closed_trades: list[Trade] = []
        self._daily_realized_pnl: float = 0.0
        self._daily_reset_day: int = self._today()
        self._peak_equity: float = initial_capital

        # Per-symbol ATR tracking (set externally from indicator pipeline)
        self._latest_atr: dict[str, float] = {}
        # Per-symbol latest price (set externally from tick pipeline)
        self._latest_price: dict[str, float] = {}
        # Per-strategy win tracking for Kelly sizing
        self._strategy_wins: dict[str, int] = defaultdict(int)
        self._strategy_trades: dict[str, int] = defaultdict(int)

        logger.info(f"[Portfolio] Initialized with capital=${initial_capital:,.2f}")

    # ── Event handlers (called by broker/execution layer) ─────────────────────

    def on_position_opened(self, position: Position) -> None:
        """Called when PaperBroker/LiveBroker opens a new position."""
        self._open_positions[position.id] = position
        # Deduct the stake from cash (stake = entry_price * size)
        stake = position.entry_price * position.size
        self._cash -= stake
        logger.debug(
            f"[Portfolio] Position {position.id[:8]} opened. "
            f"Cash remaining: ${self._cash:,.2f}"
        )

    def on_trade_closed(self, trade: Trade) -> None:
        """Called when a position is closed. Updates P&L and returns stake + profit."""
        self._open_positions.pop(trade.id, None)
        # Return stake + realized P&L to cash
        returned = trade.entry_price * trade.size + trade.realized_pnl
        self._cash += returned
        self._closed_trades.append(trade)

        # Daily P&L reset check
        today = self._today()
        if today != self._daily_reset_day:
            logger.info(
                f"[Portfolio] Day changed. Daily P&L reset from "
                f"{self._daily_realized_pnl:+.2f} to 0"
            )
            self._daily_realized_pnl = 0.0
            self._daily_reset_day = today

        self._daily_realized_pnl += trade.realized_pnl

        # Update per-strategy win tracking
        strat = trade.strategy_name or "unknown"
        self._strategy_trades[strat] += 1
        if trade.realized_pnl > 0:
            self._strategy_wins[strat] += 1

        # Update peak equity for drawdown
        current_eq = self.equity
        if current_eq > self._peak_equity:
            self._peak_equity = current_eq

        pnl_str = f"+{trade.realized_pnl:.4f}" if trade.realized_pnl >= 0 else f"{trade.realized_pnl:.4f}"
        logger.info(
            f"[Portfolio] Trade closed | PnL={pnl_str} ({trade.realized_pnl_pct:+.1%}) | "
            f"Equity=${self.equity:,.2f} | Drawdown={self.drawdown_pct:.1%}"
        )

    def update_price(self, symbol: str, price: float) -> None:
        """Update the latest known price for unrealized P&L calculation."""
        self._latest_price[symbol] = price

    def update_atr(self, symbol: str, atr: float) -> None:
        """Update the latest ATR value (used by RiskEngine for stop sizing)."""
        self._latest_atr[symbol] = atr

    # ── State getters (injected into RiskEngine) ──────────────────────────────

    @property
    def equity(self) -> float:
        """Total portfolio equity = cash + market value of all open positions."""
        position_value = sum(
            pos.entry_price * pos.size + pos.unrealized_pnl(self._latest_price.get(pos.symbol, pos.entry_price))
            for pos in self._open_positions.values()
        )
        return self._cash + position_value

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def daily_pnl_pct(self) -> float:
        """Daily realized P&L as a fraction of initial capital."""
        return self._daily_realized_pnl / self._initial_capital

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity as a positive fraction."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self.equity) / self._peak_equity)

    def get_atr(self, symbol: str) -> float | None:
        return self._latest_atr.get(symbol)

    @property
    def open_symbols(self) -> list[str]:
        """List of symbols with currently open positions (for correlation filtering)."""
        return [pos.symbol for pos in self._open_positions.values()]

    def get_strategy_win_rate(self, strategy_name: str) -> float | None:
        """Win rate [0,1] for a strategy from live trade history. None if < 10 trades."""
        total = self._strategy_trades.get(strategy_name, 0)
        if total < 10:
            return None
        wins = self._strategy_wins.get(strategy_name, 0)
        return wins / total

    # ── Summary & reporting ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a full portfolio state snapshot for logging/dashboard."""
        winning_trades = [t for t in self._closed_trades if t.realized_pnl > 0]
        total_trades = len(self._closed_trades)

        return {
            "initial_capital": self._initial_capital,
            "cash": round(self._cash, 4),
            "equity": round(self.equity, 4),
            "total_return_pct": round((self.equity - self._initial_capital) / self._initial_capital * 100, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 2),
            "drawdown_pct": round(self.drawdown_pct * 100, 2),
            "peak_equity": round(self._peak_equity, 4),
            "open_positions": self.open_position_count,
            "closed_trades": total_trades,
            "win_rate_pct": round(len(winning_trades) / total_trades * 100, 1) if total_trades else 0.0,
            "total_realized_pnl": round(sum(t.realized_pnl for t in self._closed_trades), 4),
        }

    def get_risk_getters(self) -> dict[str, Callable]:
        """
        Returns a dict of callable getters ready to inject into RiskEngine.

        Usage::
            getters = portfolio.get_risk_getters()
            risk_engine.set_portfolio_getters(**getters)
        """
        return {
            "daily_pnl_pct": lambda: self.daily_pnl_pct,
            "drawdown_pct": lambda: self.drawdown_pct,
            "open_position_count": lambda: self.open_position_count,
            "equity": lambda: self.equity,
            "atr": self.get_atr,
            "open_symbols": lambda: self.open_symbols,
            "strategy_win_rate": self.get_strategy_win_rate,
        }

    @staticmethod
    def _today() -> int:
        """Returns the current UTC day as an integer (days since epoch)."""
        return int(time.time() // 86400)
