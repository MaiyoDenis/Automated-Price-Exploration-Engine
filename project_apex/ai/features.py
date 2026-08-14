"""
Project APEX — AI Feature Engineering

Generates machine learning features from raw OHLCV data.
This ensures the exact same features are used in backtesting/training
and live trading inference.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from project_apex.indicators.oscillators import RSI, MACD
from project_apex.indicators.volatility import BollingerBands, ATR
from project_apex.indicators.trend import ADX


class FeatureGenerator:
    """
    Computes a standardized set of features for ML models.

    Results are cached: if the last row's timestamp matches the previous
    call, the cached DataFrame is returned immediately — avoiding redundant
    rolling-indicator computation on every drift check or retrain attempt.
    """

    def __init__(self) -> None:
        self.rsi = RSI(period=14)
        self.macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        self.bb = BollingerBands(period=20, num_std=2.0)
        self.atr = ATR(period=14)
        self.adx = ADX(period=14)

        # Cache: (last_timestamp, last_row_count) -> DataFrame
        self._cache_key: tuple[int | None, int] = (None, 0)
        self._cache_df: pd.DataFrame | None = None

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with [timestamp, open, high, low, close]
        and appends ML features. Does NOT drop NaNs (caller handles this).

        Returns a cached result when the input data hasn't changed since
        the last call (matched by last timestamp value + row count).
        """
        # ── Cache check ────────────────────────────────────────────────────
        if len(df) > 0:
            last_ts = int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else None
            cache_key = (last_ts, len(df))
            if cache_key == self._cache_key and self._cache_df is not None:
                return self._cache_df
        else:
            cache_key = (None, 0)

        # ── Compute features ───────────────────────────────────────────────
        df = df.copy()

        # 1. Base Indicators
        df = self.rsi.calculate(df)
        df = self.macd.calculate(df)
        df = self.bb.calculate(df)
        df = self.atr.calculate(df)
        df = self.adx.calculate(df)

        # 2. Normalized Price Features (Momentum)
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_3"] = df["close"].pct_change(3)
        df["ret_5"] = df["close"].pct_change(5)

        # Distance to Bollinger Bands: -1 at lower band, 0 at mid, +1 at upper
        if self.bb.name_pct_b in df.columns:
            df["bb_position"] = (df[self.bb.name_pct_b] * 2) - 1.0

        # Normalized Volatility (ATR / Close)
        if self.atr.name in df.columns:
            df["volatility_norm"] = df[self.atr.name] / df["close"]

        # 3. Microstructure Features
        df["hl_range_norm"] = (df["high"] - df["low"]) / df["close"]

        hl_range = df["high"] - df["low"]
        df["close_pos_in_bar"] = np.where(
            hl_range == 0, 0.5, (df["close"] - df["low"]) / hl_range
        )

        # ── Store cache ────────────────────────────────────────────────────
        self._cache_key = cache_key
        self._cache_df = df

        return df

    def invalidate_cache(self) -> None:
        """Explicitly drop the cached result (e.g. after a retrain with new data)."""
        self._cache_key = (None, 0)
        self._cache_df = None

    def get_feature_columns(self) -> list[str]:
        """Returns the list of column names used as input for the ML model."""
        return [
            self.rsi.name,
            self.macd.name,
            self.macd.name_signal,
            self.macd.name_hist,
            self.bb.name_width,
            "bb_position",
            "volatility_norm",
            self.adx.name,
            "ret_1",
            "ret_3",
            "ret_5",
            "hl_range_norm",
            "close_pos_in_bar",
        ]
