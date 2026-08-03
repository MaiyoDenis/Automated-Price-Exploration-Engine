"""
Project APEX — Trend Indicators

Indicators that measure the direction and strength of price trends.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from project_apex.indicators.base import Indicator
from project_apex.indicators.volatility import ATR


class ADX(Indicator):
    """
    Average Directional Index (ADX) with +DI and -DI.

    Measures trend strength regardless of direction. ADX > 25 = strong trend,
    ADX < 20 = weak/no trend. +DI / -DI lines indicate trend direction.

    Columns added:
        - ``adx_{period}``: ADX value (0-100 trend strength).
        - ``plus_di_{period}``: +DI (bullish directional indicator).
        - ``minus_di_{period}``: -DI (bearish directional indicator).
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"adx_{period}"
        self.name_plus_di = f"plus_di_{period}"
        self.name_minus_di = f"minus_di_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        high = data["high"]
        low = data["low"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = data["close"].shift(1)

        # True Range
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Directional movement
        plus_dm = high - prev_high
        minus_dm = prev_low - low

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # Smoothed with Wilder's EMA
        smooth_tr = tr.ewm(com=self.period - 1, min_periods=self.period).mean()
        smooth_plus = plus_dm.ewm(com=self.period - 1, min_periods=self.period).mean()
        smooth_minus = minus_dm.ewm(com=self.period - 1, min_periods=self.period).mean()

        plus_di = 100.0 * smooth_plus / smooth_tr.replace(0, np.nan)
        minus_di = 100.0 * smooth_minus / smooth_tr.replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

        data[self.name_plus_di] = plus_di
        data[self.name_minus_di] = minus_di
        data[self.name] = dx.ewm(com=self.period - 1, min_periods=self.period).mean()
        return data


class SuperTrend(Indicator):
    """
    SuperTrend Indicator.

    Combines ATR with a multiplier to form dynamic support/resistance levels.
    When price is above the band → uptrend (buy); below → downtrend (sell).

    Columns added:
        - ``supertrend_{period}_{multiplier}``: SuperTrend line value.
        - ``supertrend_dir_{period}_{multiplier}``: Direction: 1 = uptrend, -1 = downtrend.
    """

    def __init__(self, period: int = 7, multiplier: float = 3.0) -> None:
        self.period = period
        self.multiplier = multiplier
        self.name = f"supertrend_{period}_{int(multiplier)}"
        self.name_dir = f"supertrend_dir_{period}_{int(multiplier)}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Calculate ATR
        atr_ind = ATR(period=self.period)
        df = atr_ind.calculate(data.copy())
        atr = df[atr_ind.name]

        hl2 = (data["high"] + data["low"]) / 2.0
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        upper = basic_upper.copy()
        lower = basic_lower.copy()
        direction = pd.Series(1, index=data.index)
        supertrend = pd.Series(np.nan, index=data.index)

        for i in range(1, len(data)):
            # Upper band
            upper.iloc[i] = (
                min(basic_upper.iloc[i], upper.iloc[i - 1])
                if data["close"].iloc[i - 1] <= upper.iloc[i - 1]
                else basic_upper.iloc[i]
            )
            # Lower band
            lower.iloc[i] = (
                max(basic_lower.iloc[i], lower.iloc[i - 1])
                if data["close"].iloc[i - 1] >= lower.iloc[i - 1]
                else basic_lower.iloc[i]
            )
            # Direction
            if data["close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif data["close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]

        data[self.name] = supertrend
        data[self.name_dir] = direction
        return data


class ParabolicSAR(Indicator):
    """
    Parabolic Stop And Reverse (SAR).

    Tracks price with an accelerating trailing stop. Flips below price in an
    uptrend and above price in a downtrend.

    Columns added:
        - ``psar``: SAR value.
        - ``psar_bull``: True when SAR is below price (uptrend).
    """

    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> None:
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.name = "psar"
        self.name_bull = "psar_bull"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        highs = data["high"].values
        lows = data["low"].values
        n = len(data)

        sar = np.full(n, np.nan)
        bull = np.full(n, True)
        af = self.af_start
        ep = lows[0]  # extreme point
        sar[0] = highs[0]

        for i in range(1, n):
            prev_sar = sar[i - 1]
            prev_bull = bull[i - 1]

            if prev_bull:
                sar[i] = prev_sar + af * (ep - prev_sar)
                sar[i] = min(sar[i], lows[i - 1], lows[max(i - 2, 0)])
                if lows[i] < sar[i]:
                    # Reversal to bearish
                    bull[i] = False
                    sar[i] = ep
                    ep = highs[i]
                    af = self.af_start
                else:
                    bull[i] = True
                    if highs[i] > ep:
                        ep = highs[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                sar[i] = prev_sar + af * (ep - prev_sar)
                sar[i] = max(sar[i], highs[i - 1], highs[max(i - 2, 0)])
                if highs[i] > sar[i]:
                    # Reversal to bullish
                    bull[i] = True
                    sar[i] = ep
                    ep = lows[i]
                    af = self.af_start
                else:
                    bull[i] = False
                    if lows[i] < ep:
                        ep = lows[i]
                        af = min(af + self.af_step, self.af_max)

        data[self.name] = sar
        data[self.name_bull] = bull
        return data
