"""
Project APEX — Oscillator Indicators

Momentum-based oscillators that measure the speed and direction of price movements.
All indicators inherit from the Indicator base class and add new columns to the DataFrame.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from project_apex.indicators.base import Indicator


class RSI(Indicator):
    """
    Relative Strength Index (RSI).

    Measures the magnitude of recent price changes to evaluate overbought/oversold
    conditions. Ranges from 0 to 100.

    Columns added:
        - ``rsi_{period}``: RSI value.
    """

    def __init__(self, period: int = 14, column: str = "close") -> None:
        self.period = period
        self.column = column
        self.name = f"rsi_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")

        delta = data[self.column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        data[self.name] = 100.0 - (100.0 / (1.0 + rs))
        return data


class Stochastic(Indicator):
    """
    Stochastic Oscillator (%K and %D).

    Compares the closing price to a price range over a look-back period to
    identify turning points.

    Columns added:
        - ``stoch_k_{k_period}``: %K line (fast stochastic).
        - ``stoch_d_{k_period}_{d_period}``: %D signal line (smoothed %K).
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> None:
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.name = f"stoch_k_{k_period}"
        self.name_d = f"stoch_d_{k_period}_{d_period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        low_min = data["low"].rolling(window=self.k_period).min()
        high_max = data["high"].rolling(window=self.k_period).max()

        raw_k = 100.0 * (data["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        data[self.name] = raw_k.rolling(window=self.smooth_k).mean()
        data[self.name_d] = data[self.name].rolling(window=self.d_period).mean()
        return data


class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD).

    Measures the relationship between two EMAs to identify momentum shifts.

    Columns added:
        - ``macd_{fast}_{slow}``: MACD line (fast EMA − slow EMA).
        - ``macd_signal_{fast}_{slow}_{signal}``: Signal line (EMA of MACD).
        - ``macd_hist_{fast}_{slow}_{signal}``: Histogram (MACD − signal).
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
        self.name = f"macd_{fast_period}_{slow_period}"
        self.name_signal = f"macd_signal_{fast_period}_{slow_period}_{signal_period}"
        self.name_hist = f"macd_hist_{fast_period}_{slow_period}_{signal_period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")

        ema_fast = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data[self.column].ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        data[self.name] = macd_line
        data[self.name_signal] = signal_line
        data[self.name_hist] = macd_line - signal_line
        return data


class CCI(Indicator):
    """
    Commodity Channel Index (CCI).

    Measures how far price has deviated from its statistical average.
    Values above +100 suggest overbought; below -100 suggest oversold.

    Columns added:
        - ``cci_{period}``: CCI value.
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.name = f"cci_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        sma_tp = typical_price.rolling(window=self.period).mean()
        mean_dev = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        data[self.name] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
        return data


class ROC(Indicator):
    """
    Rate of Change (ROC) / Price Momentum Oscillator.

    Measures percentage change in price over a look-back period.

    Columns added:
        - ``roc_{period}``: ROC value as percentage.
    """

    def __init__(self, period: int = 10, column: str = "close") -> None:
        self.period = period
        self.column = column
        self.name = f"roc_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")

        data[self.name] = data[self.column].pct_change(periods=self.period) * 100.0
        return data
