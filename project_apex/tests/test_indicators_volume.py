"""
Project APEX — Tests for Volume Indicators

Covers: OBV, VWAP (cumulative + rolling), MFI, ChaikinMF —
column creation, monotonicity, value ranges, identity properties, error handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.indicators.volume import OBV, MFI, VWAP, ChaikinMF


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(length: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(length).cumsum()
    high = close + rng.random(length) * 2 + 0.5
    low = close - rng.random(length) * 2 - 0.5
    volume = rng.integers(100, 1000, size=length).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def _make_rising_df(length: int = 30) -> pd.DataFrame:
    """Strictly rising close prices with constant positive volume."""
    close = np.array([100.0 + i for i in range(length)])
    high = close + 1.0
    low = close - 1.0
    volume = np.full(length, 1000.0)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def _make_falling_df(length: int = 30) -> pd.DataFrame:
    """Strictly falling close prices with constant positive volume."""
    close = np.array([200.0 - i for i in range(length)])
    high = close + 1.0
    low = close - 1.0
    volume = np.full(length, 1000.0)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# OBV tests
# ---------------------------------------------------------------------------

class TestOBV:
    def test_column_added(self) -> None:
        df = _make_df()
        obv = OBV()
        result = obv.calculate(df)
        assert obv.name in result.columns

    def test_column_name_is_obv(self) -> None:
        assert OBV().name == "obv"

    def test_obv_increases_when_price_rises(self) -> None:
        """Rising close prices → OBV should be monotonically non-decreasing."""
        df = _make_rising_df()
        obv = OBV()
        result = obv.calculate(df)
        obv_series = result[obv.name]
        # Skip first row (diff is NaN → direction is 0)
        diffs = obv_series.diff().dropna()
        assert (diffs >= 0).all(), "OBV must be non-decreasing when prices rise."

    def test_obv_decreases_when_price_falls(self) -> None:
        """Falling close prices → OBV should be monotonically non-increasing."""
        df = _make_falling_df()
        obv = OBV()
        result = obv.calculate(df)
        obv_series = result[obv.name]
        diffs = obv_series.diff().dropna()
        assert (diffs <= 0).all(), "OBV must be non-increasing when prices fall."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="volume"):
            OBV().calculate(df)

    def test_no_nan_values(self) -> None:
        """OBV is cumulative and should produce no NaN values."""
        df = _make_df()
        obv = OBV()
        result = obv.calculate(df)
        assert result[obv.name].notna().all()


# ---------------------------------------------------------------------------
# VWAP tests (cumulative and rolling)
# ---------------------------------------------------------------------------

class TestVWAP:
    def test_cumulative_column_added(self) -> None:
        df = _make_df()
        vwap = VWAP(period=0)
        result = vwap.calculate(df)
        assert vwap.name in result.columns
        assert vwap.name == "vwap"

    def test_rolling_column_added(self) -> None:
        df = _make_df()
        vwap = VWAP(period=20)
        result = vwap.calculate(df)
        assert vwap.name in result.columns
        assert vwap.name == "vwap_20"

    def test_cumulative_vwap_no_nan(self) -> None:
        """Cumulative VWAP (period=0) should have no NaN (cumsum from row 0)."""
        df = _make_df()
        vwap = VWAP(period=0)
        result = vwap.calculate(df)
        assert result[vwap.name].notna().all()

    def test_rolling_vwap_warmup(self) -> None:
        """Rolling VWAP should have NaN for first period-1 rows."""
        period = 10
        df = _make_df()
        vwap = VWAP(period=period)
        result = vwap.calculate(df)
        assert result[vwap.name].iloc[: period - 1].isna().all()
        assert not np.isnan(result[vwap.name].iloc[period - 1])

    def test_vwap_close_to_typical_price(self) -> None:
        """When all prices are constant, VWAP == typical price."""
        close = 100.0
        df = pd.DataFrame({
            "high": [close] * 20,
            "low": [close] * 20,
            "close": [close] * 20,
            "volume": [1000.0] * 20,
        })
        vwap = VWAP(period=0)
        result = vwap.calculate(df)
        assert (result[vwap.name] == close).all(), "VWAP of constant price must equal that price."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            VWAP().calculate(df)


# ---------------------------------------------------------------------------
# MFI tests
# ---------------------------------------------------------------------------

class TestMFI:
    def test_column_added(self) -> None:
        df = _make_df()
        mfi = MFI(period=14)
        result = mfi.calculate(df)
        assert mfi.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert MFI(period=14).name == "mfi_14"

    def test_bounded_0_to_100(self) -> None:
        df = _make_df()
        mfi = MFI(period=14)
        result = mfi.calculate(df)
        valid = result[mfi.name].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "MFI must be in [0, 100]."

    def test_warmup_period_nan(self) -> None:
        """MFI uses rolling(period) on money flows derived from typical_price.diff().
        For random data with both positive and negative moves, the rolling window
        first produces a valid value at index period-1 (0-indexed).
        """
        period = 14
        df = _make_df(length=80)
        mfi = MFI(period=period)
        result = mfi.calculate(df)
        # Rows 0 to period-2 are NaN
        assert result[mfi.name].iloc[: period - 1].isna().all()
        # Row at index period-1 is the first non-NaN
        assert not np.isnan(result[mfi.name].iloc[period - 1])

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            MFI().calculate(df)

    def test_produces_valid_values_on_zigzag_prices(self) -> None:
        """MFI requires both positive and negative price moves to produce non-NaN values.
        Use a zigzag pattern: large up moves with small down moves → mostly positive flow."""
        period = 5
        length = 60
        # Zigzag: 3 up, 1 small down, repeat → ensures negative flow exists so MFI is computable
        close = []
        price = 100.0
        for i in range(length):
            price += 1.0 if i % 4 != 3 else -0.3
            close.append(price)
        close_arr = np.array(close)
        high = close_arr + 0.5
        low = close_arr - 0.5
        volume = np.full(length, 1000.0)
        df = pd.DataFrame({"high": high, "low": low, "close": close_arr, "volume": volume})
        mfi = MFI(period=period)
        result = mfi.calculate(df)
        valid = result[mfi.name].dropna()
        assert len(valid) > 0, "MFI should produce valid values for zigzag price series."
        # With mostly upward moves, MFI should be significantly above 50
        last = valid.iloc[-1]
        assert last > 60, f"Expected MFI above 60 for mostly-bullish price action, got {last:.1f}"


# ---------------------------------------------------------------------------
# ChaikinMF tests
# ---------------------------------------------------------------------------

class TestChaikinMF:
    def test_column_added(self) -> None:
        df = _make_df()
        cmf = ChaikinMF(period=20)
        result = cmf.calculate(df)
        assert cmf.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert ChaikinMF(period=20).name == "cmf_20"

    def test_bounded_minus1_to_1(self) -> None:
        """CMF should be in [-1, 1] by construction."""
        df = _make_df()
        cmf = ChaikinMF(period=20)
        result = cmf.calculate(df)
        valid = result[cmf.name].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all(), "CMF must be in [-1, 1]."

    def test_warmup_period_nan(self) -> None:
        period = 20
        df = _make_df()
        cmf = ChaikinMF(period=period)
        result = cmf.calculate(df)
        assert result[cmf.name].iloc[: period - 1].isna().all()

    def test_buying_pressure_positive_cmf(self) -> None:
        """Close at high end of H-L range + high volume → positive CMF."""
        length = 40
        high = np.full(length, 110.0)
        low = np.full(length, 90.0)
        close = np.full(length, 109.9)  # close near high → bullish MFM
        volume = np.full(length, 1000.0)
        df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
        cmf = ChaikinMF(period=5)
        result = cmf.calculate(df)
        valid = result[cmf.name].dropna()
        assert (valid > 0).all(), "CMF should be positive when close is near the high."

    def test_selling_pressure_negative_cmf(self) -> None:
        """Close at low end of H-L range + high volume → negative CMF."""
        length = 40
        high = np.full(length, 110.0)
        low = np.full(length, 90.0)
        close = np.full(length, 90.1)  # close near low → bearish MFM
        volume = np.full(length, 1000.0)
        df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
        cmf = ChaikinMF(period=5)
        result = cmf.calculate(df)
        valid = result[cmf.name].dropna()
        assert (valid < 0).all(), "CMF should be negative when close is near the low."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            ChaikinMF().calculate(df)
