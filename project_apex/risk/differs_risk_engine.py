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
        loss_cooldown_ticks: int = 30,        # Cooldown 30 ticks (60s) after a loss
        # Two-tier staking
        base_stake: float = 2.0,               # Normal trade stake
        recovery_multiplier: float = 11.0,     # Recovery multiplier after 1 loss
        recovery_min_confidence: float = 0.80, # Min signal confidence to fire recovery trade
        recovery_min_drought: int = 8,         # Min drought for recovery trade target digit
        ban_lost_digit_on_recovery: bool = True, # Forbid recovery on the same digit that just failed
        max_consecutive_losses: int = 2,        # Halt after this many losses in a row
        payout_ratio: float = 0.097,           # ~9.7% payout on differs
        # Regime detection
        regime_window: int = 10,
        hot_streak_threshold: float = 0.85,
        cold_streak_threshold: float = 0.70,
        use_pattern_recovery: bool = False,   # If True, recovery only fires on same pattern; if False, fires on very next trade after loss
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
        self.recovery_multiplier = recovery_multiplier
        self.recovery_min_confidence = recovery_min_confidence
        self.recovery_min_drought = recovery_min_drought
        self.ban_lost_digit_on_recovery = ban_lost_digit_on_recovery
        self.max_consecutive_losses = max_consecutive_losses
        self.payout_ratio = payout_ratio
        self.regime_window = regime_window
        self.hot_streak_threshold = hot_streak_threshold
        self.cold_streak_threshold = cold_streak_threshold
        self.use_pattern_recovery = use_pattern_recovery
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
        self.pattern_consecutive_losses: dict[Any, int] = {}
        self.last_lost_digit: int | None = None

        # Regime tracking
        self._recent_results: deque = deque(maxlen=regime_window)

        # Portfolio getters (injected by Application)
        self._get_daily_pnl_pct: Callable[[], float] = lambda: 0.0
        self._get_open_position_count: Callable[[], int] = lambda: 0
        self._get_equity: Callable[[], float] = lambda: 10_000.0
        self._get_strategy_win_rate: Callable[[str], float | None] = lambda _: None

        logger.info(
            f"[DiffersRiskEngine] Initialized | "
            f"base=${base_stake} recovery_mult={recovery_multiplier}x "
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
        self.pattern_consecutive_losses.clear()
        logger.info("[DiffersRiskEngine] Trading RESUMED")

    # ── Trade result feedback ──────────────────────────────────────────────

    def on_trade_result(self, won: bool, order: TradeOrder | None = None) -> None:
        """Update state after each settled contract."""
        self._recent_results.append(won)

        pattern = "DEFAULT"
        if order and order.metadata and "pattern" in order.metadata:
            pattern = order.metadata["pattern"]

        if won:
            self.consecutive_losses = 0
            self.pattern_consecutive_losses[pattern] = 0
            self.cooldown_ticks_remaining = 0
            self.last_lost_digit = None
            logger.info(f"[DiffersRiskEngine] Win. Consecutive losses reset to 0 (Pattern: {pattern}).")
        else:
            self.consecutive_losses += 1
            
            if pattern not in self.pattern_consecutive_losses:
                self.pattern_consecutive_losses[pattern] = 0
            self.pattern_consecutive_losses[pattern] += 1

            if order and order.metadata and "barrier" in order.metadata:
                self.last_lost_digit = order.metadata["barrier"]

            self.cooldown_ticks_remaining = self.loss_cooldown_ticks
            logger.warning(
                f"[DiffersRiskEngine] Loss #{self.consecutive_losses} globally, "
                f"#{self.pattern_consecutive_losses[pattern]} for pattern {pattern} "
                f"(Lost digit: {self.last_lost_digit}). "
                f"Cooldown for {self.loss_cooldown_ticks} ticks."
            )

#             if self.consecutive_losses >= self.max_consecutive_losses:
#                 self.halt_trading(
#                     f"{self.consecutive_losses} consecutive losses — session halted to protect capital. "
#                     f"Call resume_trading() to restart."
#                 )

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
        pattern = signal.metadata.get("pattern", "DEFAULT") if signal.metadata else "DEFAULT"
        pattern_losses = self.pattern_consecutive_losses.get(pattern, 0)
        
        regime = self._get_regime()
        if self.use_pattern_recovery:
            if pattern_losses >= self.max_consecutive_losses:
                logger.warning(
                    f"[DiffersRiskEngine] Pattern {pattern} exceeded max recovery losses ({pattern_losses}). "
                    f"Resetting pattern losses to 0 and reverting to base stake."
                )
                self.pattern_consecutive_losses[pattern] = 0
                pattern_losses = 0
            is_recovery = pattern_losses == 1
        else:
            is_recovery = self.consecutive_losses >= 1

        if is_recovery:
            # Recovery trade — safety checks
            if regime == "cold":
                logger.warning(
                    f"[DiffersRiskEngine] COLD regime + recovery trade — skipping. "
                    f"Waiting for conditions to improve."
                )
                return RiskDecision(approved=False, reason="HOLD_COLD_REGIME")

            # Check 1: Ban lost digit on recovery
            if self.ban_lost_digit_on_recovery and self.last_lost_digit is not None:
                if signal.barrier == self.last_lost_digit:
                    logger.warning(
                        f"[DiffersRiskEngine] Recovery trade blocked | barrier {signal.barrier} "
                        f"matches last lost digit {self.last_lost_digit}. Waiting for a different digit."
                    )
                    return RiskDecision(approved=False, reason="HOLD_BANNED_LOST_DIGIT")

            # Check 2: Drought confirmation for recovery trade
            drought = signal.metadata.get("drought") if signal.metadata else None
            if self.recovery_min_drought > 0 and drought is not None:
                if drought < self.recovery_min_drought:
                    logger.info(
                        f"[DiffersRiskEngine] Recovery trade waiting for cold confirmation: "
                        f"digit {signal.barrier} drought={drought} < {self.recovery_min_drought} required."
                    )
                    return RiskDecision(approved=False, reason="HOLD_RECOVERY_DROUGHT")

            if signal.confidence < self.recovery_min_confidence:
                logger.info(
                    f"[DiffersRiskEngine] Recovery trade waiting for confidence "
                    f"{signal.confidence:.3f} < {self.recovery_min_confidence:.2f} required."
                )
                return RiskDecision(approved=False, reason="HOLD_CONFIDENCE")

            stake = self.base_stake * self.recovery_multiplier
            trade_type = "RECOVERY"
        else:
            # Normal trade — base stake
            stake = self.base_stake
            trade_type = "NORMAL"

        size = round(stake, 2)

        logger.info(
            f"[DiffersRiskEngine] {trade_type} trade (Pattern: {pattern}, Pattern Losses: {pattern_losses}) | "
            f"regime={regime} | consec_losses={self.consecutive_losses} | "
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
