"""
Project APEX — Alert Manager

In-system notification hub. Publishes structured alerts on all significant
trading events. Alerts are:
  - Always logged via loguru (WARNING level for trade events, CRITICAL for halts)
  - Stored in an in-memory ring buffer accessible by the dashboard
  - Optionally extended to external channels (Telegram, email) via handlers

Alert categories:
  TRADE_OPENED   — new position entered
  TRADE_CLOSED   — position closed with P&L
  HALT_TRIGGERED — circuit breaker or risk engine halt
  HALT_LIFTED    — trading resumed after halt
  MODEL_UPDATED  — ML model retrained and improved
  REGIME_SHIFT   — market regime changed
  DAILY_SUMMARY  — end-of-day performance summary
  SYSTEM         — general system messages (connection, startup)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from loguru import logger


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertCategory(Enum):
    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    HALT_TRIGGERED = "HALT_TRIGGERED"
    HALT_LIFTED = "HALT_LIFTED"
    MODEL_UPDATED = "MODEL_UPDATED"
    REGIME_SHIFT = "REGIME_SHIFT"
    DAILY_SUMMARY = "DAILY_SUMMARY"
    SYSTEM = "SYSTEM"


@dataclass
class Alert:
    """A single structured alert."""
    category: AlertCategory
    level: AlertLevel
    title: str
    body: str
    data: dict = field(default_factory=dict)   # Machine-readable payload
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "level": self.level.value,
            "title": self.title,
            "body": self.body,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class AlertManager:
    """
    Central notification hub.

    Stores the last ``buffer_size`` alerts in a ring buffer which the dashboard
    can poll. Logs every alert through loguru at the appropriate level.

    Args:
        buffer_size: Maximum alerts to keep in memory.
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._buffer: Deque[Alert] = deque(maxlen=buffer_size)
        logger.info("[AlertManager] Initialized.")

    # ── Public emit methods ───────────────────────────────────────────────────

    def trade_opened(self, symbol: str, direction: str, size: float, price: float, strategy: str) -> None:
        self._emit(Alert(
            category=AlertCategory.TRADE_OPENED,
            level=AlertLevel.INFO,
            title=f"📈 Trade Opened — {direction} {symbol}",
            body=(
                f"Strategy: {strategy} | Size: {size:.4f} | "
                f"Entry: {price:.5f}"
            ),
            data={"symbol": symbol, "direction": direction, "size": size,
                  "price": price, "strategy": strategy},
        ))

    def trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
        strategy: str,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_str = f"+{pnl:.4f}" if pnl >= 0 else f"{pnl:.4f}"
        self._emit(Alert(
            category=AlertCategory.TRADE_CLOSED,
            level=AlertLevel.INFO,
            title=f"{emoji} Trade Closed — {direction} {symbol} | P&L: {pnl_str}",
            body=(
                f"P&L: {pnl_str} ({pnl_pct:+.2%}) | "
                f"Reason: {reason} | Strategy: {strategy}"
            ),
            data={
                "symbol": symbol, "direction": direction, "pnl": pnl,
                "pnl_pct": pnl_pct, "reason": reason, "strategy": strategy,
            },
        ))

    def halt_triggered(self, trigger: str, reason: str, halt_duration_h: float) -> None:
        self._emit(Alert(
            category=AlertCategory.HALT_TRIGGERED,
            level=AlertLevel.CRITICAL,
            title=f"🚨 TRADING HALT — {trigger}",
            body=f"{reason} | Duration: {halt_duration_h:.1f}h",
            data={"trigger": trigger, "reason": reason, "halt_duration_h": halt_duration_h},
        ))

    def halt_lifted(self) -> None:
        self._emit(Alert(
            category=AlertCategory.HALT_LIFTED,
            level=AlertLevel.WARNING,
            title="✅ Trading Resumed",
            body="Halt period has ended. System is back in active mode.",
        ))

    def model_updated(self, old_sharpe: float, new_sharpe: float, trigger: str) -> None:
        self._emit(Alert(
            category=AlertCategory.MODEL_UPDATED,
            level=AlertLevel.INFO,
            title=f"🧠 ML Model Updated (trigger={trigger})",
            body=(
                f"Validation Sharpe improved: {old_sharpe:.3f} → {new_sharpe:.3f}"
            ),
            data={"old_sharpe": old_sharpe, "new_sharpe": new_sharpe, "trigger": trigger},
        ))

    def regime_shift(self, symbol: str, old_regime: str, new_regime: str) -> None:
        self._emit(Alert(
            category=AlertCategory.REGIME_SHIFT,
            level=AlertLevel.INFO,
            title=f"🔄 Regime Shift — {symbol}",
            body=f"{old_regime} → {new_regime}",
            data={"symbol": symbol, "old_regime": old_regime, "new_regime": new_regime},
        ))

    def daily_summary(self, equity: float, return_pct: float, trades: int, win_rate: float, drawdown_pct: float) -> None:
        emoji = "📈" if return_pct >= 0 else "📉"
        self._emit(Alert(
            category=AlertCategory.DAILY_SUMMARY,
            level=AlertLevel.INFO,
            title=f"{emoji} Daily Summary | Return: {return_pct:+.2%}",
            body=(
                f"Equity: ${equity:,.2f} | Trades: {trades} | "
                f"Win Rate: {win_rate:.1%} | Max DD: {drawdown_pct:.2%}"
            ),
            data={
                "equity": equity, "return_pct": return_pct,
                "trades": trades, "win_rate": win_rate, "drawdown_pct": drawdown_pct,
            },
        ))

    def system(self, message: str, level: AlertLevel = AlertLevel.INFO) -> None:
        self._emit(Alert(
            category=AlertCategory.SYSTEM,
            level=level,
            title=f"⚙ System — {message[:60]}",
            body=message,
        ))

    # ── Dashboard access ──────────────────────────────────────────────────────

    def get_recent(self, n: int = 50) -> list[dict]:
        """Returns the N most recent alerts as serializable dicts."""
        alerts = list(self._buffer)
        alerts.reverse()
        return [a.to_dict() for a in alerts[:n]]

    def get_by_category(self, category: str, n: int = 20) -> list[dict]:
        """Filter alerts by category name."""
        try:
            cat = AlertCategory(category)
        except ValueError:
            return []
        return [
            a.to_dict() for a in reversed(list(self._buffer))
            if a.category == cat
        ][:n]

    @property
    def unread_count(self) -> int:
        """Number of alerts in the buffer (for dashboard badge)."""
        return len(self._buffer)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, alert: Alert) -> None:
        """Store alert and log it."""
        self._buffer.append(alert)
        msg = f"[{alert.category.value}] {alert.title} — {alert.body}"
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)
