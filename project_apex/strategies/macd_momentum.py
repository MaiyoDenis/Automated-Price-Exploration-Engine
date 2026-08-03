"""
Project APEX — MACD Momentum Strategy

Trades the MACD crossover with histogram confirmation and SuperTrend
direction filter to align with the dominant trend.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.indicators.oscillators import MACD
from project_apex.indicators.trend import SuperTrend


class MACDMomentumStrategy(LiveStrategy):
    """
    MACD Crossover + SuperTrend Direction Filter.

    Signal logic:
    - BUY  when MACD line crosses above signal line AND SuperTrend is bullish
    - SELL when MACD line crosses below signal line AND SuperTrend is bearish
    - Confidence scales with histogram magnitude relative to recent ATR.
    """

    _MIN_BARS = 40

    def initialize(self, config: dict[str, Any]) -> None:
        self._fast: int = config.get("macd_fast", 12)
        self._slow: int = config.get("macd_slow", 26)
        self._signal: int = config.get("macd_signal", 9)
        self._st_period: int = config.get("supertrend_period", 7)
        self._st_mult: float = config.get("supertrend_multiplier", 3.0)
        self._timeframe: int = config.get("timeframe", 300)  # 5-minute candles

        self._macd = MACD(
            fast_period=self._fast,
            slow_period=self._slow,
            signal_period=self._signal,
        )
        self._supertrend = SuperTrend(
            period=self._st_period,
            multiplier=self._st_mult,
        )

        logger.info(
            f"[{self.name}] MACD({self._fast},{self._slow},{self._signal}) "
            f"SuperTrend({self._st_period},{self._st_mult})"
        )

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if candle.timeframe != self._timeframe:
            return None

        history = self._append_candle(candle)
        if len(history) < self._MIN_BARS:
            return None

        df = self._get_history_df(candle)
        df = self._macd.calculate(df)
        df = self._supertrend.calculate(df)

        if len(df) < 2:
            return None

        cur = df.iloc[-1]
        prev = df.iloc[-2]

        macd_col = self._macd.name
        signal_col = self._macd.name_signal
        hist_col = self._macd.name_hist
        st_dir_col = self._supertrend.name_dir

        # Guard against NaNs
        for col in (macd_col, signal_col, hist_col, st_dir_col):
            if cur[col] != cur[col] or prev[col] != prev[col]:
                return None

        bullish_cross = (prev[macd_col] <= prev[signal_col]) and (cur[macd_col] > cur[signal_col])
        bearish_cross = (prev[macd_col] >= prev[signal_col]) and (cur[macd_col] < cur[signal_col])
        st_bullish = cur[st_dir_col] == 1
        st_bearish = cur[st_dir_col] == -1

        signal_type: SignalType | None = None

        if bullish_cross and st_bullish:
            signal_type = SignalType.BUY
        elif bearish_cross and st_bearish:
            signal_type = SignalType.SELL

        if signal_type is None:
            return None

        # Confidence based on histogram magnitude (normalised by recent range)
        hist_magnitude = abs(cur[hist_col])
        recent_max_hist = df[hist_col].abs().tail(20).max()
        confidence = min(1.0, hist_magnitude / (recent_max_hist + 1e-9))

        logger.debug(
            f"[{self.name}] {signal_type.name} on {candle.symbol} | "
            f"MACD={cur[macd_col]:.4f} hist={cur[hist_col]:.4f} "
            f"ST_dir={cur[st_dir_col]} conf={confidence:.2f}"
        )

        return TradeSignal(
            symbol=candle.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            metadata={
                "macd": round(cur[macd_col], 5),
                "macd_signal": round(cur[signal_col], 5),
                "macd_hist": round(cur[hist_col], 5),
                "supertrend_dir": int(cur[st_dir_col]),
                "timeframe": candle.timeframe,
            },
        )
