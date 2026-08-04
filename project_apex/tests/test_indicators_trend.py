"""
Project APEX — Tests for Trend Indicators

Covers: ADX, SuperTrend, ParabolicSAR — column creation, value ranges,
direction constraints, reversal behaviour, error handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.indicators.trend import ADX, ParabolicSAR, SuperTrend


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(length: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(length).cumsum()
    high = close + rng.random(length) * 2 + 0.5
    low = close - rng.random(length) * 2 - 0.5
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def _make_uptrend_df(length: int = 60) -> pd.DataFrame:
    """Monotonically rising prices — clear uptrend."""
    close = np.array([100.0 + i for i in range(length)])
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def _make_downtrend_df(length: int = 60) -> pd.DataFrame:
    """Monotonically falling prices — clear downtrend."""
    close = np.array([200.0 - i for i in range(length)])
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
# ADX tests
# ---------------------------------------------------------------------------

class TestADX:
    def test_three_columns_added(self) -> None:
        df = _make_df()
        adx = ADX(period=14)
        result = adx.calculate(df)
        assert adx.name in result.columns
        assert adx.name_plus_di in result.columns
        assert adx.name_minus_di in result.columns

    def test_column_names_reflect_period(self) -> None:
        adx = ADX(period=10)
        assert adx.name == "adx_10"
        assert adx.name_plus_di == "plus_di_10"
        assert adx.name_minus_di == "minus_di_10"

    def test_adx_bounded_0_to_100(self) -> None:
        df = _make_df(length=150)
        adx = ADX(period=14)
        result = adx.calculate(df)
        valid = result[adx.name].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "ADX must be in [0, 100]."

    def test_plus_di_non_negative(self) -> None:
        df = _make_df(length=150)
        adx = ADX(period=14)
        result = adx.calculate(df)
        valid = result[adx.name_plus_di].dropna()
        assert (valid >= 0).all(), "+DI must be non-negative."

    def test_minus_di_non_negative(self) -> None:
        df = _make_df(length=150)
        adx = ADX(period=14)
        result = adx.calculate(df)
        valid = result[adx.name_minus_di].dropna()
        assert (valid >= 0).all(), "-DI must be non-negative."

    def test_uptrend_plus_di_dominates(self) -> None:
        """In a strong uptrend, +DI should generally exceed -DI."""
        df = _make_uptrend_df(length=80)
        adx = ADX(period=7)
        result = adx.calculate(df)
        valid = result.dropna(subset=[adx.name_plus_di, adx.name_minus_di]).tail(20)
        if len(valid) == 0:
            pytest.skip("Not enough valid rows.")
        # At least 70% of rows should have +DI > -DI in an uptrend
        dominance_rate = (valid[adx.name_plus_di] > valid[adx.name_minus_di]).mean()
        assert dominance_rate >= 0.7, f"+DI should dominate in uptrend, got {dominance_rate:.0%}"

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="high"):
            ADX().calculate(df)


# ---------------------------------------------------------------------------
# SuperTrend tests
# ---------------------------------------------------------------------------

class TestSuperTrend:
    def test_two_columns_added(self) -> None:
        df = _make_df()
        st = SuperTrend(period=7, multiplier=3.0)
        result = st.calculate(df)
        assert st.name in result.columns
        assert st.name_dir in result.columns

    def test_column_names_reflect_params(self) -> None:
        st = SuperTrend(period=10, multiplier=2.0)
        assert st.name == "supertrend_10_2"
        assert st.name_dir == "supertrend_dir_10_2"

    def test_direction_only_1_or_minus1(self) -> None:
        df = _make_df(length=150)
        st = SuperTrend(period=7, multiplier=3.0)
        result = st.calculate(df)
        valid_dir = result[st.name_dir].dropna()
        assert set(valid_dir.unique()).issubset({1, -1}), (
            f"Direction must be only 1 or -1, got {valid_dir.unique()}"
        )

    def test_uptrend_direction_is_1(self) -> None:
        """Strong uptrend → SuperTrend direction should eventually be +1."""
        df = _make_uptrend_df(length=80)
        st = SuperTrend(period=7, multiplier=3.0)
        result = st.calculate(df)
        tail = result[st.name_dir].dropna().tail(10)
        assert (tail == 1).all(), "Should be in uptrend direction (+1)."

    def test_supertrend_value_non_nan_after_warmup(self) -> None:
        df = _make_df(length=100)
        st = SuperTrend(period=7)
        result = st.calculate(df)
        valid = result[st.name].dropna()
        assert len(valid) > 0, "SuperTrend should have non-NaN values after warmup."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="high"):
            SuperTrend().calculate(df)


# ---------------------------------------------------------------------------
# ParabolicSAR tests
# ---------------------------------------------------------------------------

class TestParabolicSAR:
    def test_two_columns_added(self) -> None:
        df = _make_df()
        psar = ParabolicSAR()
        result = psar.calculate(df)
        assert psar.name in result.columns
        assert psar.name_bull in result.columns

    def test_psar_column_name(self) -> None:
        psar = ParabolicSAR()
        assert psar.name == "psar"
        assert psar.name_bull == "psar_bull"

    def test_bull_is_boolean_series(self) -> None:
        df = _make_df()
        psar = ParabolicSAR()
        result = psar.calculate(df)
        assert result[psar.name_bull].dtype == bool

    def test_uptrend_sar_below_price(self) -> None:
        """In a stable uptrend (psar_bull=True), SAR must be below the close price.
        Skip the first few bars where SAR may not have stabilised yet."""
        df = _make_uptrend_df(length=80)
        psar = ParabolicSAR()
        result = psar.calculate(df)
        # Only check rows well after warmup where bull is True
        bull_rows = result[result[psar.name_bull]].iloc[10:]  # skip first 10 warmup bars
        if len(bull_rows) == 0:
            pytest.skip("No bullish rows found after warmup.")
        assert (bull_rows[psar.name] < bull_rows["close"]).all(), (
            "SAR must be below close in uptrend."
        )

    def test_downtrend_sar_above_price(self) -> None:
        """In a stable downtrend (psar_bull=False), SAR must be above the close price.
        Skip the first few bars where SAR may not have stabilised yet."""
        df = _make_downtrend_df(length=80)
        psar = ParabolicSAR()
        result = psar.calculate(df)
        bear_rows = result[~result[psar.name_bull]].iloc[5:]  # skip initial warmup
        if len(bear_rows) == 0:
            pytest.skip("No bearish rows found after warmup.")
        assert (bear_rows[psar.name] > bear_rows["close"]).all(), (
            "SAR must be above close in downtrend."
        )

    def test_bull_reversal_occurs(self) -> None:
        """Price that goes from downtrend to uptrend should trigger a reversal."""
        # Downtrend then uptrend
        down = list(range(100, 50, -1))
        up = list(range(50, 120))
        close = np.array(down + up, dtype=float)
        high = close + 1.0
        low = close - 1.0
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        psar = ParabolicSAR()
        result = psar.calculate(df)
        # Should have both True and False in bull column
        assert result[psar.name_bull].any(), "Should have some bullish periods."
        assert (~result[psar.name_bull]).any(), "Should have some bearish periods."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="high"):
            ParabolicSAR().calculate(df)
