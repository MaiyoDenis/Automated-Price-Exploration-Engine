"""
Project APEX — Circuit Breaker

An independent safety layer that monitors trade outcomes for dangerous patterns
and issues a HALT to the RiskEngine when protective thresholds are breached.

Triggers:
  1. Consecutive Stop-Loss Streak: 3+ consecutive SL hits → halt for rest of day
  2. Loss Velocity: Equity drops > 2% in any 30-minute window → pause 60 minutes
  3. Trade Frequency Spike: > 10 trades per hour → pause 30 minutes (prevents runaway loops)

Persistence:
  State is stored in the ``circuit_breaker_state`` table (one-row singleton).
  On startup the previous halt is restored so a process restart cannot be used
  to bypass a safety halt that is still within its cooldown window.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from loguru import logger

if TYPE_CHECKING:
    from project_apex.database.sqlite_manager import SQLiteManager


@dataclass
class CircuitBreakerEvent:
    """Emitted when a circuit breaker trigger fires."""
    trigger: str          # Name of the rule that fired
    reason: str           # Human-readable description
    halt_until: float     # Monotonic time after which trading may resume
    fired_at: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """
    Independent halt/resume controller for the RiskEngine.

    Args:
        halt_on_consecutive_sl: Halt after this many consecutive stop-losses.
        sl_lookback_s: Window (seconds) in which consecutive SLs are counted.
        loss_velocity_pct: Halt if equity drops this fraction in velocity_window_s.
        velocity_window_s: Window (seconds) for loss-velocity measurement.
        max_trades_per_hour: Halt if this many trades fire in 60 minutes.
        halt_duration_s: How long (seconds) each trigger keeps trading halted.
        on_halt: Async callback invoked when a halt fires (e.g. RiskEngine.halt_trading).
        on_resume: Async callback invoked when a halt clears.
    """

    def __init__(
        self,
        halt_on_consecutive_sl: int = 3,
        sl_lookback_s: float = 3600.0,
        loss_velocity_pct: float = 0.02,
        velocity_window_s: float = 1800.0,
        max_trades_per_hour: int = 10,
        halt_duration_s: float = 3600.0,
        on_halt: Optional[Callable[[str], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        db: Optional["SQLiteManager"] = None,
    ) -> None:
        self._sl_streak_limit = halt_on_consecutive_sl
        self._sl_lookback = sl_lookback_s
        self._loss_vel_pct = loss_velocity_pct
        self._vel_window = velocity_window_s
        self._max_trades_hr = max_trades_per_hour
        self._halt_duration = halt_duration_s
        self._on_halt = on_halt
        self._on_resume = on_resume
        self._db: Optional["SQLiteManager"] = db

        # State
        self._sl_times: list[float] = []           # Monotonic times of recent SL hits
        self._trade_times: list[float] = []         # Monotonic times of all trades
        self._equity_snapshots: list[tuple[float, float]] = []  # (time, equity)
        self._halt_until: float = 0.0              # monotonic deadline
        self._halt_until_wall: float = 0.0         # wall-clock deadline (persisted)
        self._active_trigger: str = ""

        logger.info(
            f"[CircuitBreaker] Initialized | "
            f"max_SL_streak={halt_on_consecutive_sl} | "
            f"loss_vel={loss_velocity_pct:.0%}/{velocity_window_s}s | "
            f"halt_duration={halt_duration_s}s"
        )

        # Restore any active halt from the database
        self._load_state_from_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade_opened(self) -> None:
        """Call whenever a new trade is opened."""
        now = time.monotonic()
        self._trade_times.append(now)
        # Prune old entries
        self._trade_times = [t for t in self._trade_times if now - t < 3600]
        self._check_trade_frequency()

    def record_trade_closed(self, is_stop_loss: bool, equity: float) -> None:
        """
        Call whenever a trade closes.

        Args:
            is_stop_loss: True if the trade hit the stop-loss.
            equity: Current portfolio equity after the trade.
        """
        now = time.monotonic()

        # Track stop-loss streak
        if is_stop_loss:
            self._sl_times.append(now)
        else:
            # Any win resets the consecutive SL streak
            self._sl_times.clear()

        # Prune old SL times outside lookback window
        self._sl_times = [t for t in self._sl_times if now - t < self._sl_lookback]

        # Equity snapshot for velocity check
        self._equity_snapshots.append((now, equity))
        self._equity_snapshots = [
            (t, e) for t, e in self._equity_snapshots if now - t < self._vel_window
        ]

        # Run checks
        self._check_sl_streak()
        self._check_loss_velocity()

    def update_equity(self, equity: float) -> None:
        """Update equity snapshot (call on each candle for velocity tracking)."""
        now = time.monotonic()
        self._equity_snapshots.append((now, equity))
        self._equity_snapshots = [
            (t, e) for t, e in self._equity_snapshots if now - t < self._vel_window
        ]
        self._check_loss_velocity()

    @property
    def is_halted(self) -> bool:
        """True if the circuit breaker is currently active."""
        if time.monotonic() >= self._halt_until:
            if self._active_trigger:
                # Halt has expired — auto-resume
                self._auto_resume()
            return False
        return True

    @property
    def halt_reason(self) -> str:
        return self._active_trigger

    # ── DB persistence ────────────────────────────────────────────────────────

    def _load_state_from_db(self) -> None:
        """Restore halt state from the database on startup."""
        if self._db is None:
            return
        try:
            rows = self._db.fetchall(
                "SELECT is_halted, halt_until, active_trigger, reason FROM circuit_breaker_state WHERE id = 1"
            )
            if not rows:
                return
            is_halted, halt_until_wall, active_trigger, reason = rows[0]
            if is_halted and halt_until_wall > time.time():
                # Compute remaining duration relative to the current monotonic clock
                remaining_s = halt_until_wall - time.time()
                self._halt_until = time.monotonic() + remaining_s
                self._halt_until_wall = halt_until_wall
                self._active_trigger = active_trigger
                logger.warning(
                    f"[CircuitBreaker] 🔄 Restored active halt from DB — "
                    f"trigger={active_trigger} | "
                    f"remaining={remaining_s/60:.1f}min | reason={reason}"
                )
        except Exception as exc:
            logger.error(f"[CircuitBreaker] Failed to load state from DB: {exc}")

    def _persist_state(self, is_halted: bool, reason: str = "") -> None:
        """Upsert the single-row halt state into the database."""
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                INSERT INTO circuit_breaker_state
                    (id, is_halted, halt_until, active_trigger, reason, last_updated)
                VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    is_halted      = excluded.is_halted,
                    halt_until     = excluded.halt_until,
                    active_trigger = excluded.active_trigger,
                    reason         = excluded.reason,
                    last_updated   = excluded.last_updated
                """,
                (
                    int(is_halted),
                    self._halt_until_wall,
                    self._active_trigger,
                    reason,
                    time.time(),
                ),
            )
        except Exception as exc:
            logger.error(f"[CircuitBreaker] Failed to persist state to DB: {exc}")

    def reset(self) -> None:
        """Manually clear all state (use with caution)."""
        self._sl_times.clear()
        self._trade_times.clear()
        self._equity_snapshots.clear()
        self._halt_until = 0.0
        self._active_trigger = ""
        logger.info("[CircuitBreaker] State reset manually.")

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_sl_streak(self) -> None:
        recent_sl = len(self._sl_times)
        if recent_sl >= self._sl_streak_limit:
            self._fire(
                trigger="CONSECUTIVE_SL",
                reason=(
                    f"{recent_sl} consecutive stop-losses within "
                    f"{self._sl_lookback/3600:.0f}h — trading halted."
                ),
            )

    def _check_loss_velocity(self) -> None:
        if len(self._equity_snapshots) < 2:
            return
        oldest_time, oldest_equity = self._equity_snapshots[0]
        _, latest_equity = self._equity_snapshots[-1]
        if oldest_equity <= 0:
            return
        loss_pct = (oldest_equity - latest_equity) / oldest_equity
        if loss_pct >= self._loss_vel_pct:
            window_min = (self._equity_snapshots[-1][0] - oldest_time) / 60
            self._fire(
                trigger="LOSS_VELOCITY",
                reason=(
                    f"Equity dropped {loss_pct:.1%} in {window_min:.0f} min "
                    f"(threshold={self._loss_vel_pct:.0%}) — velocity halt."
                ),
            )

    def _check_trade_frequency(self) -> None:
        if len(self._trade_times) > self._max_trades_hr:
            self._fire(
                trigger="TRADE_FREQUENCY",
                reason=(
                    f"{len(self._trade_times)} trades in last 60min "
                    f"(max={self._max_trades_hr}) — frequency halt."
                ),
            )

    def _fire(self, trigger: str, reason: str) -> None:
        if self.is_halted:
            return  # Already halted — don't stack
        self._halt_until = time.monotonic() + self._halt_duration
        self._halt_until_wall = time.time() + self._halt_duration
        self._active_trigger = trigger
        logger.critical(
            f"[CircuitBreaker] 🚨 HALT TRIGGERED [{trigger}] — {reason} | "
            f"Halt duration={self._halt_duration/3600:.1f}h"
        )
        # Immediately persist so a restart cannot bypass the halt
        self._persist_state(is_halted=True, reason=reason)
        if self._on_halt:
            self._on_halt(f"[CircuitBreaker] {trigger}: {reason}")

    def _auto_resume(self) -> None:
        logger.info(
            f"[CircuitBreaker] ✅ Halt expired [{self._active_trigger}] — trading resumed."
        )
        self._active_trigger = ""
        self._halt_until_wall = 0.0
        self._sl_times.clear()  # Reset streak after halt
        # Clear the persisted halt
        self._persist_state(is_halted=False, reason="expired")
        if self._on_resume:
            self._on_resume()
