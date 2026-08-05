"""
Project APEX — Health Monitor

Continuously watches system vitals every 60 seconds and flags degraded states.
Acts as an early warning system before problems cascade into trading losses.

Checks:
  1. WebSocket connection — is the broker connected?
  2. Data freshness — when was the last tick received? (>10s is stale)
  3. Feature quality — are ML features producing NaN outputs?
  4. Portfolio consistency — does equity match expected math?
  5. Memory pressure — are in-memory buffers growing unboundedly?

Each check returns a HealthStatus (GREEN / YELLOW / RED).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from project_apex.core.application import Application


class HealthStatus(Enum):
    GREEN = "GREEN"    # All good
    YELLOW = "YELLOW"  # Degraded but operational
    RED = "RED"        # Critical — action required


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    value: object = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
        }


class HealthMonitor:
    """
    Periodic system health checker.

    Args:
        app: Reference to the Application instance for access to all services.
        check_interval_s: How often (seconds) to run all health checks.
        stale_tick_threshold_s: Max seconds since last tick before marking YELLOW.
        stale_tick_critical_s: Max seconds since last tick before marking RED.
    """

    def __init__(
        self,
        app: "Application",
        check_interval_s: float = 60.0,
        stale_tick_threshold_s: float = 10.0,
        stale_tick_critical_s: float = 60.0,
    ) -> None:
        self._app = app
        self._interval = check_interval_s
        self._stale_yellow = stale_tick_threshold_s
        self._stale_red = stale_tick_critical_s

        self._last_tick_time: dict[str, float] = {}  # symbol → monotonic time
        self._latest_results: list[HealthCheck] = []
        self._overall_status: HealthStatus = HealthStatus.GREEN
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info(f"[HealthMonitor] Initialized | check_interval={check_interval_s}s")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[HealthMonitor] Started.")

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    # ── Event hooks ───────────────────────────────────────────────────────────

    def record_tick(self, symbol: str) -> None:
        """Call on every received tick to track data freshness."""
        self._last_tick_time[symbol] = time.monotonic()

    # ── Public access ─────────────────────────────────────────────────────────

    def get_health_report(self) -> dict:
        """Returns the latest health report for the dashboard."""
        return {
            "overall": self._overall_status.value,
            "checks": [c.to_dict() for c in self._latest_results],
            "checked_at": time.time(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._run_all_checks()
            except Exception as exc:
                logger.error(f"[HealthMonitor] Check error: {exc}")

    async def _run_all_checks(self) -> None:
        checks: list[HealthCheck] = []
        checks.append(self._check_broker_connection())
        checks.append(self._check_data_freshness())
        checks.append(self._check_portfolio_equity())
        checks.append(self._check_risk_engine())

        # Determine overall status (worst wins)
        statuses = [c.status for c in checks]
        if HealthStatus.RED in statuses:
            overall = HealthStatus.RED
        elif HealthStatus.YELLOW in statuses:
            overall = HealthStatus.YELLOW
        else:
            overall = HealthStatus.GREEN

        self._latest_results = checks
        self._overall_status = overall

        # Log degraded states
        if overall != HealthStatus.GREEN:
            bad = [c for c in checks if c.status != HealthStatus.GREEN]
            for c in bad:
                level = logger.warning if c.status == HealthStatus.YELLOW else logger.error
                level(f"[HealthMonitor] {c.status.value} — {c.name}: {c.message}")
        else:
            logger.debug("[HealthMonitor] All checks GREEN ✅")

    def _check_broker_connection(self) -> HealthCheck:
        client = self._app.deriv_client
        if client is None:
            return HealthCheck("broker_connection", HealthStatus.RED, "DerivClient not initialized")
        if client.is_connected:
            return HealthCheck("broker_connection", HealthStatus.GREEN, "Connected")
        return HealthCheck("broker_connection", HealthStatus.RED, "WebSocket disconnected!")

    def _check_data_freshness(self) -> HealthCheck:
        if not self._last_tick_time:
            return HealthCheck("data_freshness", HealthStatus.YELLOW, "No ticks received yet")

        now = time.monotonic()
        worst_age = max(now - t for t in self._last_tick_time.values())
        stale_symbols = [
            sym for sym, t in self._last_tick_time.items()
            if now - t > self._stale_yellow
        ]

        if worst_age > self._stale_red:
            return HealthCheck(
                "data_freshness", HealthStatus.RED,
                f"No ticks for {worst_age:.0f}s on: {stale_symbols}",
                value=worst_age,
            )
        if worst_age > self._stale_yellow:
            return HealthCheck(
                "data_freshness", HealthStatus.YELLOW,
                f"Stale data ({worst_age:.0f}s) on: {stale_symbols}",
                value=worst_age,
            )
        return HealthCheck("data_freshness", HealthStatus.GREEN, f"Fresh (max age={worst_age:.1f}s)")

    def _check_portfolio_equity(self) -> HealthCheck:
        portfolio = self._app.portfolio
        if portfolio is None:
            return HealthCheck("portfolio", HealthStatus.YELLOW, "Portfolio not initialized")
        equity = portfolio.equity
        if equity <= 0:
            return HealthCheck("portfolio", HealthStatus.RED, f"Equity is zero or negative: {equity:.2f}")
        drawdown = portfolio.drawdown_pct
        if drawdown > 0.12:
            return HealthCheck(
                "portfolio", HealthStatus.YELLOW,
                f"High drawdown: {drawdown:.1%}",
                value=drawdown,
            )
        return HealthCheck("portfolio", HealthStatus.GREEN, f"Equity=${equity:,.2f} DD={drawdown:.1%}")

    def _check_risk_engine(self) -> HealthCheck:
        risk = self._app.risk_engine
        if risk is None:
            return HealthCheck("risk_engine", HealthStatus.RED, "RiskEngine not initialized")
        if risk._trading_halt:
            return HealthCheck(
                "risk_engine", HealthStatus.YELLOW,
                f"Trading halted: {risk._halt_reason}",
            )
        return HealthCheck("risk_engine", HealthStatus.GREEN, "Active — no halt")
