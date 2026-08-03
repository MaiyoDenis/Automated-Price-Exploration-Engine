"""
Project APEX — RSI Mean Reversion Strategy

Buys when RSI is oversold (below threshold) and sells when overbought (above threshold).
Combines RSI with ADX to only trade when a meaningful trend exists, filtering out
range-bound noise.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.indicators.oscillators import RSI
from project_apex.indicators.trend import ADX


class RSIMeanReversionStrategy(LiveStrategy):
    """
    RSI Mean Reversion with ADX trend filter.

    Signal logic:
    - BUY  when RSI < oversold_threshold AND ADX >= min_adx (trend is real)
    - SELL when RSI > overbought_threshold AND ADX >= min_adx
    - Confidence is proportional to how extreme the RSI reading is.

    Default parameters are tuned for synthetic indices (Deriv R_25/R_50).
    """

    _MIN_BARS = 30  # Minimum candles needed before emitting signals

    def initialize(self, config: dict[str, Any]) -> None:
        self._rsi_period: int = config.get("rsi_period", 14)
        self._oversold: float = config.get("oversold", 30.0)
        self._overbought: float = config.get("overbought", 70.0)
        self._adx_period: int = config.get("adx_period", 14)
        self._min_adx: float = config.get("min_adx", 20.0)
        self._timeframe: int = config.get("timeframe", 60)  # 1-minute candles

        self._rsi = RSI(period=self._rsi_period)
        self._adx = ADX(period=self._adx_period)

        logger.info(
            f"[{self.name}] RSI({self._rsi_period}) "
            f"oversold={self._oversold} overbought={self._overbought} "
            f"ADX_min={self._min_adx}"
        )

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if candle.timeframe != self._timeframe:
            return None  # Only process target timeframe

        history = self._append_candle(candle)
        if len(history) < self._MIN_BARS:
            return None

        df = self._get_history_df(candle)
        df = self._rsi.calculate(df)
        df = self._adx.calculate(df)

        latest = df.iloc[-1]
        rsi_val = latest[self._rsi.name]
        adx_val = latest[self._adx.name]

        if not (rsi_val == rsi_val and adx_val == adx_val):  # NaN check
            return None

        # ADX filter — skip if market is ranging
        if adx_val < self._min_adx:
            return None

        signal_type: SignalType | None = None
        confidence: float = 0.0

        if rsi_val < self._oversold:
            signal_type = SignalType.BUY
            # Deeper oversold → higher confidence
            confidence = min(1.0, (self._oversold - rsi_val) / self._oversold)

        elif rsi_val > self._overbought:
            signal_type = SignalType.SELL
            # More overbought → higher confidence
            confidence = min(1.0, (rsi_val - self._overbought) / (100 - self._overbought))

        if signal_type is None:
            return None

        logger.debug(
            f"[{self.name}] {signal_type.name} signal on {candle.symbol} | "
            f"RSI={rsi_val:.1f} ADX={adx_val:.1f} conf={confidence:.2f}"
        )

        return TradeSignal(
            symbol=candle.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            metadata={
                "rsi": round(rsi_val, 2),
                "adx": round(adx_val, 2),
                "timeframe": candle.timeframe,
            },
        )
