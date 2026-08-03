"""
Project APEX — Volatility Indicators

Measures that capture the degree of price variation over time.
High volatility often precedes trend reversals or breakouts.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from project_apex.indicators.base import Indicator


class BollingerBands(Indicator):
    """
    Bollinger Bands.

    Plots bands at standard deviation multiples above and below a moving average.
    When price touches the bands, it often signals a reversal or continuation.

    Columns added:
        - ``bb_middle_{period}``: Middle band (SMA).
        - ``bb_upper_{period}``: Upper band (SMA + k * std).
        - ``bb_lower_{period}``: Lower band (SMA - k * std).
        - ``bb_pct_b_{period}``: %B — where price sits within the bands (0=lower, 1=upper).
        - ``bb_width_{period}``: Bandwidth (upper - lower) / middle.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        column: str = "close",
    ) -> None:
        self.period = period
        self.num_std = num_std
        self.column = column
        self.name = f"bb_middle_{period}"
        self.name_upper = f"bb_upper_{period}"
        self.name_lower = f"bb_lower_{period}"
        self.name_pct_b = f"bb_pct_b_{period}"
        self.name_width = f"bb_width_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")

        middle = data[self.column].rolling(window=self.period).mean()
        std = data[self.column].rolling(window=self.period).std()
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        band_range = (upper - lower).replace(0, np.nan)

        data[self.name] = middle
        data[self.name_upper] = upper
        data[self.name_lower] = lower
        data[self.name_pct_b] = (data[self.column] - lower) / band_range
        data[self.name_width] = band_range / middle.replace(0, np.nan)
        return data


class ATR(Indicator):
    """
    Average True Range (ATR).

    Measures market volatility as the exponential moving average of the True Range.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    Columns added:
        - ``atr_{period}``: ATR value.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"atr_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        data[self.name] = tr.ewm(com=self.period - 1, min_periods=self.period).mean()
        return data


class KeltnerChannels(Indicator):
    """
    Keltner Channels.

    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    Breakouts beyond the channel often signal strong trends.

    Columns added:
        - ``kc_middle_{period}``: Middle (EMA of close).
        - ``kc_upper_{period}``: Upper band (EMA + multiplier * ATR).
        - ``kc_lower_{period}``: Lower band (EMA - multiplier * ATR).
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> None:
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.name = f"kc_middle_{ema_period}"
        self.name_upper = f"kc_upper_{ema_period}"
        self.name_lower = f"kc_lower_{ema_period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Compute ATR inline
        atr_indicator = ATR(period=self.atr_period)
        data = atr_indicator.calculate(data)
        atr_col = atr_indicator.name

        ema = data["close"].ewm(span=self.ema_period, adjust=False).mean()
        data[self.name] = ema
        data[self.name_upper] = ema + self.multiplier * data[atr_col]
        data[self.name_lower] = ema - self.multiplier * data[atr_col]
        return data


class DonchianChannels(Indicator):
    """
    Donchian Channels.

    Plots the highest high and lowest low over a look-back period.
    Classic breakout trading: buy when price breaks the upper channel.

    Columns added:
        - ``dc_upper_{period}``: Highest high over period.
        - ``dc_lower_{period}``: Lowest low over period.
        - ``dc_middle_{period}``: Midpoint of the channel.
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.name = f"dc_upper_{period}"
        self.name_lower = f"dc_lower_{period}"
        self.name_middle = f"dc_middle_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        upper = data["high"].rolling(window=self.period).max()
        lower = data["low"].rolling(window=self.period).min()

        data[self.name] = upper
        data[self.name_lower] = lower
        data[self.name_middle] = (upper + lower) / 2.0
        return data
