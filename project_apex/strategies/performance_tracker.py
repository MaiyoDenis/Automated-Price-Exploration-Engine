"""
Project APEX — Strategy Performance Tracker

Tracks rolling live-trade performance for each individual strategy.
Used by MultiStrategyEnsemble to adaptively weight voting:
  - Winners get boosted confidence multiplier
  - Losers get reduced multiplier
  - Chronically losing strategies are temporarily disabled

Persistence:
  Historical statistics are loaded from and saved to the ``strategy_performance``
  table in SQLite so adaptive weights survive process restarts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Deque, Optional

from loguru import logger

if TYPE_CHECKING:
    from project_apex.database.sqlite_manager import SQLiteManager


@dataclass
class TradeResult:
    """A single closed trade result for a strategy."""
    pnl: float            # Absolute P&L
    pnl_pct: float        # P&L as fraction of stake
    won: bool             # True if pnl > 0


class PerformanceTracker:
    """
    Rolling performance statistics for a single strategy.

    Tracks the last ``window`` closed trades and computes:
    - Win rate
    - Expectancy (mean P&L per trade)
    - Rolling Sharpe approximation
    - Max consecutive losing streak

    The ``weight`` property returns a multiplier [0.0 → 1.5] to scale
    the strategy's confidence in the ensemble vote.

    Args:
        strategy_name: Name for logging.
        window: Number of recent trades to keep in the rolling window.
        disable_threshold: Expectancy below this value for ``disable_after``
            consecutive trades causes the strategy to be flagged as disabled.
        disable_after: Consecutive trades with bad expectancy before disabling.
    """

    def __init__(
        self,
        strategy_name: str,
        window: int = 50,
        disable_threshold: float = -0.005,
        disable_after: int = 20,
        db: Optional["SQLiteManager"] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self._window = window
        self._disable_threshold = disable_threshold
        self._disable_after = disable_after
        self._db: Optional["SQLiteManager"] = db

        self._trades: Deque[TradeResult] = deque(maxlen=window)
        self._consecutive_bad: int = 0
        self.is_disabled: bool = False

        # Restore historical stats from the database if available
        self._load_from_db()

    # ── Record a result ───────────────────────────────────────────────────────

    def record(self, pnl: float, pnl_pct: float) -> None:
        """
        Register a new closed trade result.

        Args:
            pnl: Absolute realized profit or loss.
            pnl_pct: P&L as fraction of the entry stake.
        """
        result = TradeResult(pnl=pnl, pnl_pct=pnl_pct, won=pnl > 0)
        self._trades.append(result)

        # Check if expectancy is chronically bad
        if self.expectancy < self._disable_threshold:
            self._consecutive_bad += 1
        else:
            self._consecutive_bad = 0

        if self._consecutive_bad >= self._disable_after and not self.is_disabled:
            self.is_disabled = True
            logger.warning(
                f"[PerformanceTracker] ⚠ {self.strategy_name} DISABLED — "
                f"expectancy={self.expectancy:.4f} for {self._consecutive_bad} trades. "
                f"Will re-enable when expectancy recovers."
            )

        # Auto-re-enable if expectancy recovers
        if self.is_disabled and self.expectancy > 0:
            self.is_disabled = False
            logger.info(
                f"[PerformanceTracker] ✓ {self.strategy_name} RE-ENABLED — "
                f"expectancy={self.expectancy:.4f}"
            )

        # Persist updated stats so adaptive weights survive restarts
        self._save_to_db()

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        """Fraction of winning trades [0, 1]. Returns 0.5 if no data."""
        if not self._trades:
            return 0.5
        wins = sum(1 for t in self._trades if t.won)
        return wins / len(self._trades)

    @property
    def expectancy(self) -> float:
        """Mean P&L per trade (in fraction terms). Positive = viable strategy."""
        if not self._trades:
            return 0.0
        return sum(t.pnl_pct for t in self._trades) / len(self._trades)

    @property
    def rolling_sharpe(self) -> float:
        """Approximate rolling Sharpe using trade-level returns."""
        if len(self._trades) < 5:
            return 0.0
        returns = [t.pnl_pct for t in self._trades]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        if std_r == 0:
            return 0.0
        return mean_r / std_r  # Simplified (no annualisation)

    @property
    def max_losing_streak(self) -> int:
        """Longest consecutive losing run in the window."""
        streak = current = 0
        for t in self._trades:
            if not t.won:
                current += 1
                streak = max(streak, current)
            else:
                current = 0
        return streak

    @property
    def weight(self) -> float:
        """
        Confidence multiplier for ensemble voting [0.1 → 1.5].

        Logic:
          - No data → neutral weight 1.0
          - Win rate > 60% AND expectancy > 0.5% → boost to 1.5
          - Win rate < 40% OR expectancy < 0% → reduce to 0.25
          - Otherwise → linear interpolation
        """
        if self.is_disabled:
            return 0.0
        if self.trade_count < 10:
            return 1.0  # Not enough data yet — neutral

        wr = self.win_rate
        exp = self.expectancy

        if wr >= 0.60 and exp >= 0.005:
            return 1.5
        if wr < 0.40 or exp < 0.0:
            return 0.25

        # Linear interpolation between 0.40→0.60 win rate
        base_weight = 0.25 + ((wr - 0.40) / 0.20) * 1.25
        return round(min(1.5, max(0.1, base_weight)), 3)

    def summary(self) -> dict:
        """Serializable summary for the dashboard."""
        return {
            "strategy": self.strategy_name,
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate * 100, 1),
            "expectancy_pct": round(self.expectancy * 100, 3),
            "rolling_sharpe": round(self.rolling_sharpe, 3),
            "max_losing_streak": self.max_losing_streak,
            "weight": self.weight,
            "disabled": self.is_disabled,
        }

    # ── DB persistence ────────────────────────────────────────────────────────

    def _load_from_db(self) -> None:
        """Seed the rolling window with historical stats from the database."""
        if self._db is None:
            return
        try:
            rows = self._db.fetchall(
                "SELECT wins, total_trades, expectancy, win_rate, rolling_sharpe "
                "FROM strategy_performance WHERE strategy_name = ?",
                (self.strategy_name,),
            )
            if not rows:
                return
            wins, total_trades, expectancy, win_rate, rolling_sharpe = rows[0]
            if total_trades > 0:
                # Reconstruct synthetic TradeResult entries to prime the deque
                # so win_rate and expectancy approximate the persisted values
                wins_count = int(wins)
                losses_count = int(total_trades) - wins_count
                # Add winning trades
                for _ in range(min(wins_count, self._window // 2)):
                    self._trades.append(TradeResult(pnl=abs(expectancy), pnl_pct=abs(expectancy), won=True))
                # Add losing trades
                for _ in range(min(losses_count, self._window // 2)):
                    self._trades.append(TradeResult(pnl=-abs(expectancy), pnl_pct=-abs(expectancy), won=False))
                logger.info(
                    f"[PerformanceTracker] Loaded history for {self.strategy_name} — "
                    f"trades={total_trades} | win_rate={win_rate:.1%} | "
                    f"expectancy={expectancy:.4f}"
                )
        except Exception as exc:
            logger.error(f"[PerformanceTracker] Failed to load from DB: {exc}")

    def _save_to_db(self) -> None:
        """Upsert current rolling stats to the database."""
        if self._db is None:
            return
        wins = sum(1 for t in self._trades if t.won)
        try:
            self._db.execute(
                """
                INSERT INTO strategy_performance
                    (strategy_name, wins, total_trades, expectancy, win_rate, rolling_sharpe)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_name) DO UPDATE SET
                    wins           = excluded.wins,
                    total_trades   = excluded.total_trades,
                    expectancy     = excluded.expectancy,
                    win_rate       = excluded.win_rate,
                    rolling_sharpe = excluded.rolling_sharpe
                """,
                (
                    self.strategy_name,
                    wins,
                    self.trade_count,
                    self.expectancy,
                    self.win_rate,
                    self.rolling_sharpe,
                ),
            )
        except Exception as exc:
            logger.error(f"[PerformanceTracker] Failed to save to DB: {exc}")
