"""
Project APEX — Bollinger Band Breakout Strategy

Detects Bollinger Band squeeze (low volatility) and then trades the breakout.
Uses ATR to size confidence and %B to confirm position within bands.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.indicators.volatility import BollingerBands, ATR


class BollingerBreakoutStrategy(LiveStrategy):
    """
    Bollinger Band Squeeze + Breakout.

    Signal logic:
    - Detects a squeeze when BB width < squeeze_threshold.
    - BUY  when price breaks above the upper band after a squeeze.
    - SELL when price breaks below the lower band after a squeeze.
    - Confidence scales with %B (proximity to band extremes).
    """

    _MIN_BARS = 30

    def initialize(self, config: dict[str, Any]) -> None:
        self._period: int = config.get("bb_period", 20)
        self._num_std: float = config.get("bb_num_std", 2.0)
        self._squeeze_threshold: float = config.get("squeeze_threshold", 0.02)
        self._atr_period: int = config.get("atr_period", 14)
        self._timeframe: int = config.get("timeframe", 900)  # 15-minute candles

        self._bb = BollingerBands(period=self._period, num_std=self._num_std)
        self._atr = ATR(period=self._atr_period)
        self._in_squeeze: dict[str, bool] = {}

        logger.info(
            f"[{self.name}] BB({self._period},{self._num_std}) "
            f"squeeze_threshold={self._squeeze_threshold}"
        )

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if candle.timeframe != self._timeframe:
            return None

        history = self._append_candle(candle)
        if len(history) < self._MIN_BARS:
            return None

        df = self._get_history_df(candle)
        df = self._bb.calculate(df)
        df = self._atr.calculate(df)

        cur = df.iloc[-1]

        bb_upper = cur[self._bb.name_upper]
        bb_lower = cur[self._bb.name_lower]
        bb_width = cur[self._bb.name_width]
        pct_b = cur[self._bb.name_pct_b]
        price = candle.close

        # Guard NaNs
        for v in (bb_upper, bb_lower, bb_width, pct_b):
            if v != v:
                return None

        symbol = candle.symbol
        was_in_squeeze = self._in_squeeze.get(symbol, False)

        # Detect squeeze
        if bb_width < self._squeeze_threshold:
            self._in_squeeze[symbol] = True
            return None  # Wait for breakout

        # Breakout after squeeze
        signal_type: SignalType | None = None
        confidence: float = 0.0

        if was_in_squeeze:
            if price > bb_upper:
                signal_type = SignalType.BUY
                confidence = min(1.0, max(0.0, pct_b - 1.0) + 0.5)  # pct_b > 1 → above upper band
            elif price < bb_lower:
                signal_type = SignalType.SELL
                confidence = min(1.0, max(0.0, -pct_b) + 0.5)      # pct_b < 0 → below lower band

        self._in_squeeze[symbol] = False  # Reset squeeze state after breakout attempt

        if signal_type is None:
            return None

        confidence = max(0.1, min(1.0, confidence))

        logger.debug(
            f"[{self.name}] {signal_type.name} on {symbol} | "
            f"BB_width={bb_width:.4f} %B={pct_b:.2f} conf={confidence:.2f}"
        )

        return TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            metadata={
                "bb_upper": round(bb_upper, 5),
                "bb_lower": round(bb_lower, 5),
                "bb_width": round(bb_width, 5),
                "pct_b": round(pct_b, 3),
                "was_in_squeeze": was_in_squeeze,
                "timeframe": candle.timeframe,
            },
        )
