"""
Project APEX — Tests for Moving Average Indicators

Covers: SMA, EMA — column creation, warmup period, value correctness,
missing-column error handling, and relative-speed behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.indicators.moving_averages import EMA, SMA


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(length: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(length).cumsum()
    high = close + rng.random(length) * 2
    low = close - rng.random(length) * 2
    volume = rng.integers(100, 1000, size=length).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# SMA tests
# ---------------------------------------------------------------------------

class TestSMA:
    def test_column_added(self) -> None:
        df = _make_df()
        sma = SMA(period=10)
        result = sma.calculate(df)
        assert sma.name in result.columns, f"Expected column '{sma.name}' not found."

    def test_column_name_reflects_period(self) -> None:
        sma = SMA(period=20)
        assert sma.name == "sma_20"

    def test_warmup_period_first_n_minus_one_nan(self) -> None:
        """First period-1 rows must be NaN; row at index period-1 must be valid."""
        period = 10
        df = _make_df(length=50)
        sma = SMA(period=period)
        result = sma.calculate(df)
        col = result[sma.name]
        # All rows before the window is full should be NaN
        assert col.iloc[: period - 1].isna().all(), "Pre-warmup rows should be NaN."
        assert not np.isnan(col.iloc[period - 1]), "Row at index period-1 should be valid."

    def test_value_matches_manual_rolling_mean(self) -> None:
        """SMA value must equal pandas rolling mean on the same data."""
        period = 5
        df = _make_df(length=30)
        sma = SMA(period=period)
        result = sma.calculate(df.copy())
        expected = df["close"].rolling(window=period).mean()
        pd.testing.assert_series_equal(result[sma.name], expected, check_names=False)

    def test_custom_column(self) -> None:
        """SMA with column='high' should operate on the high column."""
        period = 5
        df = _make_df(length=30)
        sma = SMA(period=period, column="high")
        result = sma.calculate(df.copy())
        expected = df["high"].rolling(window=period).mean()
        pd.testing.assert_series_equal(result[sma.name], expected, check_names=False)

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        sma = SMA(period=2)
        with pytest.raises(ValueError, match="close"):
            sma.calculate(df)

    def test_does_not_mutate_original_by_default(self) -> None:
        """calculate() should add the column to the passed DataFrame in-place."""
        df = _make_df()
        original_columns = set(df.columns)
        sma = SMA(period=5)
        result = sma.calculate(df)
        # In-place: result IS df
        assert sma.name in result.columns
        assert sma.name in df.columns  # confirms in-place mutation
        assert original_columns | {sma.name} == set(df.columns)


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestEMA:
    def test_column_added(self) -> None:
        df = _make_df()
        ema = EMA(period=10)
        result = ema.calculate(df)
        assert ema.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        ema = EMA(period=21)
        assert ema.name == "ema_21"

    def test_no_nan_after_first_row(self) -> None:
        """EMA (adjust=False) should be defined from the first row onward."""
        df = _make_df(length=50)
        ema = EMA(period=10)
        result = ema.calculate(df)
        # With adjust=False, pandas EMA produces values for all rows
        assert result[ema.name].notna().all(), "EMA should have no NaN values."

    def test_value_matches_pandas_ewm(self) -> None:
        period = 10
        df = _make_df(length=40)
        ema = EMA(period=period)
        result = ema.calculate(df.copy())
        expected = df["close"].ewm(span=period, adjust=False).mean()
        pd.testing.assert_series_equal(result[ema.name], expected, check_names=False)

    def test_ema_reacts_faster_than_sma_after_spike(self) -> None:
        """After a large price spike, EMA should be closer to spike than SMA."""
        period = 10
        close = [100.0] * 20 + [200.0] * 5  # sudden spike
        df = pd.DataFrame({"close": close})
        sma = SMA(period=period)
        ema = EMA(period=period)
        df = sma.calculate(df)
        df = ema.calculate(df)
        # At the spike peak, EMA > SMA (EMA weights recent prices more)
        assert df[ema.name].iloc[-1] > df[sma.name].iloc[-1], (
            "EMA should react faster than SMA to the spike."
        )

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        ema = EMA(period=2)
        with pytest.raises(ValueError, match="close"):
            ema.calculate(df)

    def test_custom_column(self) -> None:
        period = 5
        df = _make_df(length=30)
        ema = EMA(period=period, column="volume")
        result = ema.calculate(df.copy())
        expected = df["volume"].ewm(span=period, adjust=False).mean()
        pd.testing.assert_series_equal(result[ema.name], expected, check_names=False)
