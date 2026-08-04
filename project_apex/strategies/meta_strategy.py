"""
Project APEX — Meta Strategy (Regime-Aware)

A top-level Strategy that detects the market regime and routes events
only to the appropriate underlying sub-strategies.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
import pandas as pd

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal
from project_apex.ai.regime import RegimeDetector, MarketRegime
from project_apex.indicators.trend import ADX
from project_apex.indicators.volatility import ATR


class MetaRegimeStrategy(LiveStrategy):
    """
    Acts as a router. Determines the current market regime, and only 
    asks strategies suited for that regime to evaluate the candle.
    """

    def __init__(
        self,
        trend_strategies: list[LiveStrategy],
        ranging_strategies: list[LiveStrategy],
        name: str = "MetaRegimeRouter",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.trend_strategies = trend_strategies
        self.ranging_strategies = ranging_strategies
        super().__init__(name=name, config=config or {})

    def initialize(self, config: dict[str, Any]) -> None:
        self._timeframe: int = config.get("timeframe", 60)
        self.detector = RegimeDetector()
        
        # We need ADX and ATR for the regime detector
        self.adx = ADX(period=14)
        self.atr = ATR(period=14)
        
        self.current_regime = MarketRegime.RANGING

        logger.info(
            f"[{self.name}] Initialized | "
            f"Trend Strategies: {[s.name for s in self.trend_strategies]} | "
            f"Ranging Strategies: {[s.name for s in self.ranging_strategies]}"
        )

    def on_tick(self, tick: Tick) -> TradeSignal | None:
        """Route tick to active strategies."""
        # Depending on complexity, we might route to all or only active.
        # For safety, we route to all so they can maintain internal state if needed.
        for s in self.trend_strategies + self.ranging_strategies:
            s.on_tick(tick)
        return None

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if candle.timeframe != self._timeframe:
            return None

        # 1. Update internal regime history
        history = self._append_candle(candle)
        if len(history) < 50:
            return None

        df = self._get_history_df(candle)
        df = self.adx.calculate(df)
        df = self.atr.calculate(df)

        # 2. Detect Regime
        new_regime = self.detector.detect(
            df,
            adx_col=self.adx.name,
            plus_di_col=self.adx.name_plus_di,
            minus_di_col=self.adx.name_minus_di,
            atr_col=self.atr.name
        )
        
        if new_regime != self.current_regime:
            logger.info(f"[{self.name}] Regime Shift: {self.current_regime.value} ──► {new_regime.value}")
            self.current_regime = new_regime

        # 3. Route Candle to appropriate strategies
        active_strategies = []
        
        if self.current_regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
            active_strategies = self.trend_strategies
        elif self.current_regime == MarketRegime.RANGING:
            active_strategies = self.ranging_strategies
        elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
            # Maybe we don't trade during extreme volatility
            return None

        # 4. Return the first valid signal (or use an ensemble mechanism here)
        # For simplicity, returning the first non-null signal. 
        # In practice, this could wrap a MultiStrategyEnsemble per regime.
        for strategy in active_strategies:
            sig = strategy.on_candle(candle)
            if sig is not None:
                # Add regime info to metadata
                sig.metadata['regime'] = self.current_regime.value
                return sig

        # Ensure inactive strategies still process the candle to update their internal indicators
        inactive = [s for s in (self.trend_strategies + self.ranging_strategies) if s not in active_strategies]
        for strategy in inactive:
            strategy.on_candle(candle)

        return None
