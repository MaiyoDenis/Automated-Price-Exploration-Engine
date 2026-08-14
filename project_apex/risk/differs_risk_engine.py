"""
Project APEX — Differs Risk Engine

Recovery staking strategy:
  - Normal trade:    stake = base_stake (e.g. $2)
  - Recovery trade:  stake = recovery_stake (e.g. $22) — fired only after 1 loss
                     AND only when signal confidence >= recovery_min_confidence
  - After 2 consecutive losses: halt trading for the session — do NOT martingale further

Regime detection (rolling 10-trade window):
  - HOT  (≥85% win rate) → allowed to trade normally
  - COLD (<70% win rate) → suppresses recovery trade, cuts to base stake only
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Awaitable, Any

from loguru import logger

from project_apex.strategies.signals import TradeSignal
from project_apex.strategies.differs_signal import DifferSignal
from project_apex.risk.models import TradeOrder, RiskDecision, Direction

OrderHandler = Callable[[TradeOrder], Awaitable[None]]


class DiffersRiskEngine:
    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        loss_cooldown_ticks: int = 5,
        # Two-tier staking
        base_stake: float = 2.0,               # Normal trade stake
        recovery_stake: float = 22.0,          # Recovery stake after 1 loss
        recovery_min_confidence: float = 0.80, # Min signal confidence to fire recovery trade
        max_consecutive_losses: int = 2,        # Halt after this many losses in a row
        payout_ratio: float = 0.097,           # ~9.7% payout on differs
        # Regime detection
        regime_window: int = 10,
        hot_streak_threshold: float = 0.85,
        cold_streak_threshold: float = 0.70,
        # Legacy / unused but kept for dashboard compatibility
        kelly_fraction: float = 0.0,
        max_stake: float = 50.0,
        martingale_enabled: bool = False,
        martingale_multiplier: float = 1.0,
        martingale_max_steps: int = 1,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.loss_cooldown_ticks = loss_cooldown_ticks
        self.base_stake = base_stake
        self.recovery_stake = recovery_stake
        self.recovery_min_confidence = recovery_min_confidence
        self.max_consecutive_losses = max_consecutive_losses
        self.payout_ratio = payout_ratio
        self.regime_window = regime_window
        self.hot_streak_threshold = hot_streak_threshold
        self.cold_streak_threshold = cold_streak_threshold
        # Legacy compatibility
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake
        self.martingale_enabled = martingale_enabled
        self.martingale_multiplier = martingale_multiplier
        self.martingale_max_steps = martingale_max_steps

        # State
        self._trading_halt: bool = False
        self._halt_reason: str = ""
        self._order_handlers: list[OrderHandler] = []

        self.cooldown_ticks_remaining: int = 0
        self.consecutive_losses: int = 0

        # Regime tracking
        self._recent_results: deque = deque(maxlen=regime_window)

        # Portfolio getters (injected by Application)
        self._get_daily_pnl_pct: Callable[[], float] = lambda: 0.0
        self._get_open_position_count: Callable[[], int] = lambda: 0
        self._get_equity: Callable[[], float] = lambda: 10_000.0
        self._get_strategy_win_rate: Callable[[str], float | None] = lambda _: None

        logger.info(
            f"[DiffersRiskEngine] Initialized | "
            f"base=${base_stake} recovery=${recovery_stake} "
            f"recovery_min_conf={recovery_min_confidence:.0%} "
            f"max_consec_losses={max_consecutive_losses} "
            f"cooldown={loss_cooldown_ticks}t daily_loss={max_daily_loss_pct:.0%}"
        )

    # ── Dependency injection ───────────────────────────────────────────────

    def set_portfolio_getters(
        self,
        daily_pnl_pct: Callable[[], float],
        open_position_count: Callable[[], int],
        equity: Callable[[], float],
        strategy_win_rate: Callable[[str], float | None] | None = None,
    ) -> None:
        self._get_daily_pnl_pct = daily_pnl_pct
        self._get_open_position_count = open_position_count
        self._get_equity = equity
        if strategy_win_rate:
            self._get_strategy_win_rate = strategy_win_rate

    def add_order_handler(self, handler: OrderHandler) -> None:
        self._order_handlers.append(handler)

    # ── Halt control ───────────────────────────────────────────────────────

    def halt_trading(self, reason: str) -> None:
        self._trading_halt = True
        self._halt_reason = reason
        logger.warning(f"[DiffersRiskEngine] Trading HALTED: {reason}")

    def resume_trading(self) -> None:
        self._trading_halt = False
        self._halt_reason = ""
        self.consecutive_losses = 0
        logger.info("[DiffersRiskEngine] Trading RESUMED")

    # ── Trade result feedback ──────────────────────────────────────────────

    def on_trade_result(self, won: bool) -> None:
        """Update state after each settled contract."""
        self._recent_results.append(won)

        if won:
            self.consecutive_losses = 0
            self.cooldown_ticks_remaining = 0
            logger.info("[DiffersRiskEngine] Win. Consecutive losses reset to 0.")
        else:
            self.consecutive_losses += 1
            self.cooldown_ticks_remaining = self.loss_cooldown_ticks
            logger.warning(
                f"[DiffersRiskEngine] Loss #{self.consecutive_losses}. "
                f"Cooldown for {self.loss_cooldown_ticks} ticks."
            )

            if self.consecutive_losses >= self.max_consecutive_losses:
                self.halt_trading(
                    f"{self.consecutive_losses} consecutive losses — session halted to protect capital. "
                    f"Call resume_trading() to restart."
                )

    def on_tick(self) -> None:
        """Decrement cooldown on every tick."""
        if self.cooldown_ticks_remaining > 0:
            self.cooldown_ticks_remaining -= 1
            if self.cooldown_ticks_remaining == 0:
                logger.info("[DiffersRiskEngine] Cooldown finished. Ready to trade.")

    # ── Regime detection ───────────────────────────────────────────────────

    def _get_regime(self) -> str:
        """
        HOT    ≥ hot_streak_threshold  → trade normally
        NORMAL between thresholds      → trade normally
        COLD   < cold_streak_threshold → skip recovery trade, base stake only
        """
        if len(self._recent_results) < self.regime_window // 2:
            return "normal"

        win_rate = sum(self._recent_results) / len(self._recent_results)
        if win_rate >= self.hot_streak_threshold:
            return "hot"
        elif win_rate < self.cold_streak_threshold:
            return "cold"
        return "normal"

    # ── Signal evaluation ─────────────────────────────────────────────────

    async def evaluate(self, signal: TradeSignal) -> None:
        if not isinstance(signal, DifferSignal):
            return

        decision = self._evaluate_sync(signal)

        if decision.approved and decision.order is not None:
            logger.info(
                f"[DiffersRiskEngine] ✓ APPROVED {signal.strategy_name} "
                f"{signal.symbol} | digit={signal.barrier} | "
                f"stake=${decision.order.size:.2f} | {decision.reason}"
            )
            for handler in self._order_handlers:
                await handler(decision.order)
        else:
            if decision.reason != "HOLD":
                logger.debug(
                    f"[DiffersRiskEngine] ✗ REJECTED {signal.strategy_name} "
                    f"{signal.symbol} | reason={decision.reason}"
                )

    def _evaluate_sync(self, signal: DifferSignal) -> RiskDecision:
        # Rule 1: Halt check
        if self._trading_halt:
            return RiskDecision(approved=False, reason=f"Trading halted: {self._halt_reason}")

        # Rule 2: Cooldown
        if self.cooldown_ticks_remaining > 0:
            return RiskDecision(approved=False, reason="HOLD")

        # Rule 3: Daily loss limit
        daily_pnl = self._get_daily_pnl_pct()
        if daily_pnl < -self.max_daily_loss_pct:
            self.halt_trading(
                f"Daily loss limit {self.max_daily_loss_pct:.0%} breached ({daily_pnl:.1%})"
            )
            return RiskDecision(approved=False, reason=self._halt_reason)

        # Rule 4: Max 1 open position
        if self._get_open_position_count() >= 1:
            return RiskDecision(approved=False, reason="Already in a digit trade")

        # Rule 5: Determine stake based on loss state and regime
        regime = self._get_regime()
        is_recovery = self.consecutive_losses == 1

        if is_recovery:
            # Recovery trade — only fire if confidence is high enough AND regime not cold
            if regime == "cold":
                logger.warning(
                    f"[DiffersRiskEngine] COLD regime + recovery trade — skipping. "
                    f"Waiting for conditions to improve."
                )
                return RiskDecision(approved=False, reason="HOLD")

            if signal.confidence < self.recovery_min_confidence:
                logger.info(
                    f"[DiffersRiskEngine] Recovery trade waiting for confidence "
                    f"{signal.confidence:.3f} < {self.recovery_min_confidence:.2f} required."
                )
                return RiskDecision(approved=False, reason="HOLD")

            stake = self.recovery_stake
            trade_type = "RECOVERY"
        else:
            # Normal trade — base stake
            stake = self.base_stake
            trade_type = "NORMAL"

        size = round(stake, 2)

        logger.info(
            f"[DiffersRiskEngine] {trade_type} trade | regime={regime} | "
            f"consec_losses={self.consecutive_losses} | "
            f"conf={signal.confidence:.3f} | stake=${size:.2f}"
        )

        order = TradeOrder(
            symbol=signal.symbol,
            direction=Direction.LONG,
            size=size,
            entry_price=signal.price,
            stop_loss=0.0,
            take_profit=0.0,
            strategy_name=signal.strategy_name,
            signal_confidence=signal.confidence,
            metadata={
                **signal.metadata,
                "barrier": signal.barrier,
                "duration_ticks": signal.duration_ticks,
                "trade_type": trade_type,
                "regime": regime,
                "consecutive_losses": self.consecutive_losses,
            },
        )

        return RiskDecision(
            approved=True,
            reason=f"{trade_type} | regime={regime} | Stake=${size:.2f}",
            order=order,
        )
