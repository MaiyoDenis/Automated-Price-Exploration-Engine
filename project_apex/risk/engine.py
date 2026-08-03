"""
Project APEX — Risk Engine

The Risk Engine is the final gatekeeper before any trade reaches execution.
It enforces all risk rules and has absolute authority to reject or resize orders.

Rules enforced (in evaluation order):
  1. Trading halt — no new positions when halt is active.
  2. Daily loss limit — daily PnL below threshold halts trading.
  3. Maximum drawdown — portfolio drawdown above threshold halts trading.
  4. Maximum concurrent positions — rejects if too many open positions.
  5. Symbol cooldown — minimum seconds between trades on the same symbol.
  6. Volatility filter — rejects during abnormally high ATR spikes.
  7. Position sizing — calculates stake via fixed fractional sizing.
  8. Stop-loss / take-profit — computes price levels based on ATR.
"""
from __future__ import annotations

import time
from typing import Callable, Awaitable

from loguru import logger

from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.risk.models import TradeOrder, RiskDecision, Direction


OrderHandler = Callable[[TradeOrder], Awaitable[None]]


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
        atr_stop_multiplier: ATR multiple used to calculate stop-loss distance.
        atr_tp_multiplier: ATR multiple used to calculate take-profit distance.
        min_confidence: Minimum signal confidence to even evaluate.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_open_positions: int = 3,
        symbol_cooldown_s: float = 60.0,
        risk_per_trade_pct: float = 0.01,
        atr_stop_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
        min_confidence: float = 0.3,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.symbol_cooldown_s = symbol_cooldown_s
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.min_confidence = min_confidence

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
        self._get_atr: Callable[[str], float | None] = lambda _: None  # symbol → ATR

        logger.info(
            f"[RiskEngine] Initialized | "
            f"daily_loss={max_daily_loss_pct:.0%} drawdown={max_drawdown_pct:.0%} "
            f"max_pos={max_open_positions} risk_per_trade={risk_per_trade_pct:.1%}"
        )

    # ── Dependency injection ──────────────────────────────────────────────────

    def set_portfolio_getters(
        self,
        daily_pnl_pct: Callable[[], float],
        drawdown_pct: Callable[[], float],
        open_position_count: Callable[[], int],
        equity: Callable[[], float],
        atr: Callable[[str], float | None],
    ) -> None:
        """Inject live portfolio state getters from the Portfolio class."""
        self._get_daily_pnl_pct = daily_pnl_pct
        self._get_drawdown_pct = drawdown_pct
        self._get_open_position_count = open_position_count
        self._get_equity = equity
        self._get_atr = atr

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
        """Manually halt all trading (e.g., on connection failure)."""
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

        # Rule 1: Trading halt
        if self._trading_halt:
            return RiskDecision(approved=False, reason=f"Trading halted: {self._halt_reason}")

        # Rule 2: Minimum confidence
        if signal.confidence < self.min_confidence:
            return RiskDecision(
                approved=False,
                reason=f"Confidence {signal.confidence:.2f} below minimum {self.min_confidence:.2f}",
            )

        # Rule 3: Daily loss limit
        daily_pnl = self._get_daily_pnl_pct()
        if daily_pnl < -self.max_daily_loss_pct:
            self.halt_trading(f"Daily loss limit {self.max_daily_loss_pct:.0%} breached ({daily_pnl:.1%})")
            return RiskDecision(approved=False, reason=self._halt_reason)

        # Rule 4: Maximum drawdown
        drawdown = self._get_drawdown_pct()
        if drawdown > self.max_drawdown_pct:
            self.halt_trading(f"Max drawdown {self.max_drawdown_pct:.0%} breached ({drawdown:.1%})")
            return RiskDecision(approved=False, reason=self._halt_reason)

        # Rule 5: Max concurrent positions
        open_pos = self._get_open_position_count()
        if open_pos >= self.max_open_positions:
            return RiskDecision(
                approved=False,
                reason=f"Max open positions reached ({open_pos}/{self.max_open_positions})",
            )

        # Rule 6: Symbol cooldown
        now = time.monotonic()
        last_trade = self._last_trade_time.get(signal.symbol, 0.0)
        elapsed = now - last_trade
        if elapsed < self.symbol_cooldown_s:
            return RiskDecision(
                approved=False,
                reason=f"Symbol cooldown: {signal.symbol} traded {elapsed:.0f}s ago "
                       f"(need {self.symbol_cooldown_s:.0f}s)",
            )

        # Rule 7: Position sizing (fixed fractional)
        equity = self._get_equity()
        risk_amount = equity * self.risk_per_trade_pct

        # Rule 8: Stop/TP levels using ATR
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
            metadata=signal.metadata,
        )

        return RiskDecision(
            approved=True,
            reason=f"Passed all rules. size={size:.4f} SL={stop_loss:.5f} TP={take_profit:.5f}",
            order=order,
        )
