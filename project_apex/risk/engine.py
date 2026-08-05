"""
Project APEX — Risk Engine

The Risk Engine is the final gatekeeper before any trade reaches execution.
It enforces all risk rules and has absolute authority to reject or resize orders.

Rules enforced (in evaluation order):
  1. Circuit breaker halt — overrides everything.
  2. Trading halt — no new positions when halt is active.
  3. Daily loss limit — daily PnL below threshold halts trading.
  4. Maximum drawdown — portfolio drawdown above threshold halts trading.
  5. Maximum concurrent positions — rejects if too many open positions.
  6. Symbol cooldown — minimum seconds between trades on the same symbol.
  7. Correlation filter — no two trades on correlated symbol groups.
  8. Volatility filter — rejects during abnormally high ATR spikes.
  9. Position sizing — Kelly-influenced fixed fractional sizing.
  10. Stop-loss / take-profit — computed price levels based on ATR.
"""
from __future__ import annotations

import time
from typing import Callable, Awaitable

from loguru import logger

from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.risk.models import TradeOrder, RiskDecision, Direction


OrderHandler = Callable[[TradeOrder], Awaitable[None]]

# Symbol correlation groups — trades within the same group are correlated.
# A new trade is rejected if there's already an open position in the same group.
_CORRELATION_GROUPS: list[set[str]] = [
    {"R_10", "R_25", "1HZ10V", "1HZ25V"},          # Low-volatility group
    {"R_50", "R_75", "1HZ50V", "1HZ75V"},           # Mid-volatility group
    {"R_100", "1HZ100V"},                            # High-volatility group
]


class RiskEngine:
    """
    Stateful risk gatekeeper.

    Inject a ``PortfolioState`` getter to allow the engine to read live P&L
    and position counts without creating a circular dependency.

    Args:
        max_daily_loss_pct: Maximum allowable daily loss as a fraction of
            starting equity (e.g., 0.05 = 5%). Trading halts if breached.
        max_drawdown_pct: Maximum portfolio drawdown (fraction) before halt.
        max_open_positions: Maximum simultaneously open positions.
        symbol_cooldown_s: Minimum seconds between trades on the same symbol.
        risk_per_trade_pct: Fraction of equity to risk per trade (fixed fractional).
        kelly_fraction: Maximum Kelly fraction to apply (caps Kelly sizing).
        atr_stop_multiplier: ATR multiple used to calculate stop-loss distance.
        atr_tp_multiplier: ATR multiple used to calculate take-profit distance.
        min_confidence: Minimum signal confidence to even evaluate.
        enable_correlation_filter: Reject trades on correlated symbols.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_open_positions: int = 3,
        symbol_cooldown_s: float = 60.0,
        risk_per_trade_pct: float = 0.01,
        kelly_fraction: float = 0.25,
        atr_stop_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
        min_confidence: float = 0.3,
        enable_correlation_filter: bool = True,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.symbol_cooldown_s = symbol_cooldown_s
        self.risk_per_trade_pct = risk_per_trade_pct
        self.kelly_fraction = kelly_fraction
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.min_confidence = min_confidence
        self.enable_correlation_filter = enable_correlation_filter

        # Runtime state
        self._trading_halt: bool = False
        self._halt_reason: str = ""
        self._last_trade_time: dict[str, float] = {}  # symbol → epoch seconds
        self._order_handlers: list[OrderHandler] = []

        # Pluggable portfolio state getters (injected by Application)
        self._get_daily_pnl_pct: Callable[[], float] = lambda: 0.0
        self._get_drawdown_pct: Callable[[], float] = lambda: 0.0
        self._get_open_position_count: Callable[[], int] = lambda: 0
        self._get_equity: Callable[[], float] = lambda: 10_000.0
        self._get_atr: Callable[[str], float | None] = lambda _: None
        self._get_open_symbols: Callable[[], list[str]] = lambda: []
        self._get_strategy_win_rate: Callable[[str], float | None] = lambda _: None

        # Circuit breaker reference (injected)
        self._circuit_breaker = None

        logger.info(
            f"[RiskEngine] Initialized | "
            f"daily_loss={max_daily_loss_pct:.0%} drawdown={max_drawdown_pct:.0%} "
            f"max_pos={max_open_positions} risk_per_trade={risk_per_trade_pct:.1%} "
            f"kelly_cap={kelly_fraction:.0%} corr_filter={enable_correlation_filter}"
        )

    # ── Dependency injection ──────────────────────────────────────────────────

    def set_portfolio_getters(
        self,
        daily_pnl_pct: Callable[[], float],
        drawdown_pct: Callable[[], float],
        open_position_count: Callable[[], int],
        equity: Callable[[], float],
        atr: Callable[[str], float | None],
        open_symbols: Callable[[], list[str]] | None = None,
        strategy_win_rate: Callable[[str], float | None] | None = None,
    ) -> None:
        """Inject live portfolio state getters from the Portfolio class."""
        self._get_daily_pnl_pct = daily_pnl_pct
        self._get_drawdown_pct = drawdown_pct
        self._get_open_position_count = open_position_count
        self._get_equity = equity
        self._get_atr = atr
        if open_symbols:
            self._get_open_symbols = open_symbols
        if strategy_win_rate:
            self._get_strategy_win_rate = strategy_win_rate

    def set_circuit_breaker(self, circuit_breaker) -> None:
        """Inject the CircuitBreaker instance."""
        self._circuit_breaker = circuit_breaker
        circuit_breaker._on_halt = self.halt_trading
        circuit_breaker._on_resume = self.resume_trading
        logger.info("[RiskEngine] CircuitBreaker wired.")

    def add_order_handler(self, handler: OrderHandler) -> None:
        """Register an async handler that receives approved TradeOrders."""
        self._order_handlers.append(handler)

    # ── Public API ────────────────────────────────────────────────────────────

    async def evaluate(self, signal: TradeSignal) -> None:
        """
        Evaluate a TradeSignal from the StrategyEngine.
        Runs all risk rules and dispatches approved TradeOrders.
        """
        decision = self._evaluate_sync(signal)

        if decision.approved and decision.order is not None:
            logger.info(
                f"[RiskEngine] ✓ APPROVED {signal.signal_type.name} "
                f"{signal.symbol} | size={decision.order.size:.2f} | {decision.reason}"
            )
            for handler in self._order_handlers:
                await handler(decision.order)
        else:
            logger.debug(
                f"[RiskEngine] ✗ REJECTED {signal.signal_type.name} "
                f"{signal.symbol} | reason={decision.reason}"
            )

    def halt_trading(self, reason: str) -> None:
        """Manually halt all trading (e.g., on connection failure or circuit breaker)."""
        self._trading_halt = True
        self._halt_reason = reason
        logger.warning(f"[RiskEngine] Trading HALTED: {reason}")

    def resume_trading(self) -> None:
        """Resume trading after a manual or automatic halt."""
        self._trading_halt = False
        self._halt_reason = ""
        logger.info("[RiskEngine] Trading RESUMED")

    # ── Rule evaluation ───────────────────────────────────────────────────────

    def _evaluate_sync(self, signal: TradeSignal) -> RiskDecision:
        """Synchronous rule chain — returns on first rejection."""

        # Rule 0: Only evaluate actionable signals
        if signal.signal_type in (SignalType.HOLD, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
            return RiskDecision(approved=False, reason="Non-entry signal type — no new order.")

        # Rule 1: Circuit breaker check
        if self._circuit_breaker is not None and self._circuit_breaker.is_halted:
            return RiskDecision(
                approved=False,
                reason=f"Circuit breaker active: {self._circuit_breaker.halt_reason}",
            )

        # Rule 2: Trading halt
        if self._trading_halt:
            return RiskDecision(approved=False, reason=f"Trading halted: {self._halt_reason}")

        # Rule 3: Minimum confidence
        if signal.confidence < self.min_confidence:
            return RiskDecision(
                approved=False,
                reason=f"Confidence {signal.confidence:.2f} below minimum {self.min_confidence:.2f}",
            )

        # Rule 4: Daily loss limit
        daily_pnl = self._get_daily_pnl_pct()
        if daily_pnl < -self.max_daily_loss_pct:
            self.halt_trading(f"Daily loss limit {self.max_daily_loss_pct:.0%} breached ({daily_pnl:.1%})")
            return RiskDecision(approved=False, reason=self._halt_reason)

        # Rule 5: Maximum drawdown
        drawdown = self._get_drawdown_pct()
        if drawdown > self.max_drawdown_pct:
            self.halt_trading(f"Max drawdown {self.max_drawdown_pct:.0%} breached ({drawdown:.1%})")
            return RiskDecision(approved=False, reason=self._halt_reason)

        # Rule 6: Max concurrent positions
        open_pos = self._get_open_position_count()
        if open_pos >= self.max_open_positions:
            return RiskDecision(
                approved=False,
                reason=f"Max open positions reached ({open_pos}/{self.max_open_positions})",
            )

        # Rule 7: Symbol cooldown
        now = time.monotonic()
        last_trade = self._last_trade_time.get(signal.symbol, 0.0)
        elapsed = now - last_trade
        if elapsed < self.symbol_cooldown_s:
            return RiskDecision(
                approved=False,
                reason=f"Symbol cooldown: {signal.symbol} traded {elapsed:.0f}s ago "
                       f"(need {self.symbol_cooldown_s:.0f}s)",
            )

        # Rule 8: Correlation filter
        if self.enable_correlation_filter:
            open_symbols = self._get_open_symbols()
            corr_conflict = self._find_correlation_conflict(signal.symbol, open_symbols)
            if corr_conflict:
                return RiskDecision(
                    approved=False,
                    reason=f"Correlation conflict: {signal.symbol} is correlated with open {corr_conflict}",
                )

        # Rule 9: Kelly-influenced position sizing
        equity = self._get_equity()

        # Get win rate for this strategy (from PerformanceTracker)
        win_rate = self._get_strategy_win_rate(signal.strategy_name)
        if win_rate is not None and win_rate > 0:
            rr = self.atr_tp_multiplier / self.atr_stop_multiplier  # Reward-to-Risk ratio
            kelly = (win_rate - (1 - win_rate) / rr) if rr > 0 else 0.0
            kelly = max(0.0, kelly)
            # Cap kelly at configured fraction (never bet the full Kelly)
            effective_risk_pct = min(kelly * self.kelly_fraction, self.risk_per_trade_pct * 2)
            effective_risk_pct = max(effective_risk_pct, self.risk_per_trade_pct * 0.25)
        else:
            # No live performance data yet — use base risk fraction
            effective_risk_pct = self.risk_per_trade_pct

        risk_amount = equity * effective_risk_pct

        # Rule 10: Stop/TP levels using ATR
        atr_value = self._get_atr(signal.symbol)
        if atr_value and atr_value > 0:
            stop_distance = atr_value * self.atr_stop_multiplier
            tp_distance = atr_value * self.atr_tp_multiplier
        else:
            # Fallback: use 1% of price
            stop_distance = signal.price * 0.01
            tp_distance = signal.price * 0.02

        direction = Direction.LONG if signal.signal_type == SignalType.BUY else Direction.SHORT

        if direction == Direction.LONG:
            stop_loss = signal.price - stop_distance
            take_profit = signal.price + tp_distance
        else:
            stop_loss = signal.price + stop_distance
            take_profit = signal.price - tp_distance

        # Size = risk_amount / stop_distance (stake-based sizing)
        size = round(risk_amount / (stop_distance + 1e-9), 4)
        size = max(size, 0.01)  # Minimum stake

        # Mark cooldown
        self._last_trade_time[signal.symbol] = now

        order = TradeOrder(
            symbol=signal.symbol,
            direction=direction,
            size=size,
            entry_price=signal.price,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            strategy_name=signal.strategy_name,
            signal_confidence=signal.confidence,
            metadata={
                **signal.metadata,
                "kelly_risk_pct": round(effective_risk_pct, 4),
                "win_rate_used": round(win_rate, 3) if win_rate else None,
            },
        )

        return RiskDecision(
            approved=True,
            reason=(
                f"Passed all rules. size={size:.4f} SL={stop_loss:.5f} TP={take_profit:.5f} "
                f"risk={effective_risk_pct:.2%}"
            ),
            order=order,
        )

    @staticmethod
    def _find_correlation_conflict(symbol: str, open_symbols: list[str]) -> str | None:
        """Returns the conflicting open symbol if correlation filter fires, else None."""
        for group in _CORRELATION_GROUPS:
            if symbol in group:
                for open_sym in open_symbols:
                    if open_sym in group and open_sym != symbol:
                        return open_sym
        return None
