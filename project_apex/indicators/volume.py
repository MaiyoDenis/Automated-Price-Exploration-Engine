"""
Project APEX — Volume Indicators

Volume-based indicators that confirm price moves and detect institutional activity.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from project_apex.indicators.base import Indicator


class OBV(Indicator):
    """
    On-Balance Volume (OBV).

    Cumulative volume metric. Rising OBV on rising price confirms an uptrend;
    divergence between OBV and price often predicts reversals.

    Columns added:
        - ``obv``: Cumulative on-balance volume.
    """

    def __init__(self) -> None:
        self.name = "obv"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("close", "volume"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        direction = np.sign(data["close"].diff())
        obv = (direction * data["volume"]).fillna(0).cumsum()
        data[self.name] = obv
        return data


class VWAP(Indicator):
    """
    Volume Weighted Average Price (VWAP).

    Represents the average price weighted by volume. Intraday traders use it as
    a benchmark — price above VWAP is bullish, below is bearish.

    Note: VWAP is typically reset at the start of each trading session. For
    continuous tick data, this implementation uses a rolling window instead.

    Columns added:
        - ``vwap_{period}``: Rolling VWAP value (or cumulative if period=0).
    """

    def __init__(self, period: int = 0) -> None:
        """
        Args:
            period: Rolling window size. 0 = cumulative (session-style VWAP).
        """
        self.period = period
        self.name = f"vwap_{period}" if period > 0 else "vwap"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close", "volume"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        pv = typical_price * data["volume"]

        if self.period > 0:
            rolling_pv = pv.rolling(window=self.period).sum()
            rolling_vol = data["volume"].rolling(window=self.period).sum()
            data[self.name] = rolling_pv / rolling_vol.replace(0, np.nan)
        else:
            data[self.name] = pv.cumsum() / data["volume"].cumsum().replace(0, np.nan)

        return data


class MFI(Indicator):
    """
    Money Flow Index (MFI).

    A volume-weighted RSI that measures the flow of money into and out of an asset.
    MFI > 80 = overbought; MFI < 20 = oversold.

    Columns added:
        - ``mfi_{period}``: MFI value (0-100).
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.name = f"mfi_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close", "volume"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        raw_money_flow = typical_price * data["volume"]

        price_diff = typical_price.diff()

        positive_flow = raw_money_flow.where(price_diff > 0, 0.0)
        negative_flow = raw_money_flow.where(price_diff < 0, 0.0)

        pos_sum = positive_flow.rolling(window=self.period).sum()
        neg_sum = negative_flow.rolling(window=self.period).sum()

        mfr = pos_sum / neg_sum.replace(0, np.nan)
        data[self.name] = 100.0 - (100.0 / (1.0 + mfr))
        return data


class ChaikinMF(Indicator):
    """
    Chaikin Money Flow (CMF).

    Measures money flow volume over a look-back period. CMF > 0 = buying pressure;
    CMF < 0 = selling pressure.

    Columns added:
        - ``cmf_{period}``: CMF value (-1 to +1).
    """

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.name = f"cmf_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close", "volume"):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        hl_range = (data["high"] - data["low"]).replace(0, np.nan)
        money_flow_multiplier = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / hl_range
        money_flow_volume = money_flow_multiplier * data["volume"]

        cmf = (
            money_flow_volume.rolling(window=self.period).sum()
            / data["volume"].rolling(window=self.period).sum().replace(0, np.nan)
        )
        data[self.name] = cmf
        return data
