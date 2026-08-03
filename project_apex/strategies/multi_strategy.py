"""
Project APEX — Multi-Strategy Ensemble

Combines signals from multiple LiveStrategy instances using a confidence-weighted
voting mechanism. Emits a final signal only when enough strategies agree.
This is the primary strategy used in live trading.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType


class MultiStrategyEnsemble(LiveStrategy):
    """
    Ensemble (voting) strategy.

    Each registered sub-strategy votes BUY, SELL, or HOLD.
    Votes are weighted by confidence. A final signal is emitted only when
    the weighted vote for one direction exceeds ``min_vote_fraction``.

    This effectively requires multiple strategies to agree before a trade,
    dramatically reducing false positives.

    Args:
        strategies: List of ``LiveStrategy`` instances to consult.
        min_vote_fraction: Fraction of total confidence that must agree
            (e.g., 0.6 means 60% of combined confidence must back one direction).
        min_strategies_agree: Minimum number of strategies that must independently
            emit the same direction for a signal to fire.
    """

    def __init__(
        self,
        strategies: list[LiveStrategy],
        min_vote_fraction: float = 0.55,
        min_strategies_agree: int = 2,
        name: str = "MultiStrategyEnsemble",
        config: dict[str, Any] | None = None,
    ) -> None:
        self._sub_strategies = strategies
        self._min_vote_fraction = min_vote_fraction
        self._min_strategies_agree = min_strategies_agree
        super().__init__(name=name, config=config or {})

    def initialize(self, config: dict[str, Any]) -> None:
        logger.info(
            f"[{self.name}] Ensemble with {len(self._sub_strategies)} strategies | "
            f"min_vote={self._min_vote_fraction:.0%} "
            f"min_agree={self._min_strategies_agree}"
        )

    def on_tick(self, tick: Tick) -> TradeSignal | None:
        """Forward tick to all sub-strategies (most will return None)."""
        for strategy in self._sub_strategies:
            strategy.on_tick(tick)
        return None  # Ensemble only aggregates candle signals

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        """
        Collect signals from all sub-strategies and apply ensemble voting.
        """
        signals: list[TradeSignal] = []
        for strategy in self._sub_strategies:
            sig = strategy.on_candle(candle)
            if sig is not None and sig.signal_type != SignalType.HOLD:
                signals.append(sig)

        if not signals:
            return None

        # Bucket votes by signal direction (BUY / SELL)
        vote_confidence: dict[SignalType, float] = defaultdict(float)
        vote_count: dict[SignalType, int] = defaultdict(int)

        for sig in signals:
            vote_confidence[sig.signal_type] += sig.confidence
            vote_count[sig.signal_type] += 1

        total_confidence = sum(vote_confidence.values())
        if total_confidence == 0:
            return None

        # Find dominant direction
        dominant = max(vote_confidence, key=lambda k: vote_confidence[k])
        dom_fraction = vote_confidence[dominant] / total_confidence
        dom_count = vote_count[dominant]

        if dom_fraction < self._min_vote_fraction:
            logger.debug(
                f"[{self.name}] No consensus on {candle.symbol}: "
                f"{dominant.name}={dom_fraction:.0%} (need {self._min_vote_fraction:.0%})"
            )
            return None

        if dom_count < self._min_strategies_agree:
            logger.debug(
                f"[{self.name}] Not enough strategies agree on {candle.symbol}: "
                f"{dom_count} (need {self._min_strategies_agree})"
            )
            return None

        # Aggregate confidence = weighted average
        ensemble_confidence = vote_confidence[dominant] / len(self._sub_strategies)
        ensemble_confidence = min(1.0, ensemble_confidence)

        contributing = [s.strategy_name for s in signals if s.signal_type == dominant]
        logger.info(
            f"[{self.name}] ✓ CONSENSUS {dominant.name} on {candle.symbol} | "
            f"fraction={dom_fraction:.0%} strategies={contributing} "
            f"confidence={ensemble_confidence:.2f}"
        )

        return TradeSignal(
            symbol=candle.symbol,
            signal_type=dominant,
            confidence=ensemble_confidence,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            metadata={
                "contributing_strategies": contributing,
                "vote_fraction": round(dom_fraction, 3),
                "vote_count": dom_count,
                "all_signals": [
                    {"strategy": s.strategy_name, "type": s.signal_type.name, "conf": s.confidence}
                    for s in signals
                ],
            },
        )
