"""
Project APEX — Tests for Oscillator Indicators

Covers: RSI, Stochastic, MACD, CCI, ROC — column creation, boundary conditions,
warmup periods, mathematical identities, and error handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.indicators.oscillators import CCI, MACD, ROC, RSI, Stochastic


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(length: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(length).cumsum()
    high = close + rng.random(length) * 2
    low = close - rng.random(length) * 2
    volume = rng.integers(100, 1000, size=length).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_column_added(self) -> None:
        df = _make_df()
        rsi = RSI(period=14)
        result = rsi.calculate(df)
        assert rsi.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert RSI(period=7).name == "rsi_7"

    def test_bounded_0_to_100(self) -> None:
        df = _make_df()
        rsi = RSI(period=14)
        result = rsi.calculate(df)
        valid = result[rsi.name].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]."

    def test_warmup_period_nan(self) -> None:
        """RSI: diff() loses row 0, so EWM with min_periods=period
        produces first valid value at index period (0-based), not period-1."""
        period = 14
        df = _make_df(length=100)
        rsi = RSI(period=period)
        result = rsi.calculate(df)
        # Rows 0 to period-1 must be NaN (period rows consumed by diff + ewm warmup)
        assert result[rsi.name].iloc[: period].isna().all()
        # Row at index 'period' is the first non-NaN
        assert not np.isnan(result[rsi.name].iloc[period])

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="close"):
            RSI(period=2).calculate(df)

    def test_all_gains_rsi_near_100(self) -> None:
        """Mostly rising prices (small dips) → RSI should be very high (>80).
        Note: pure gain series give avg_loss=0 → RSI is NaN, so we use mostly-up prices."""
        period = 5
        close = []
        price = 100.0
        for i in range(60):
            price += 1.0 if i % 10 != 9 else -0.05  # mostly up, tiny dip every 10th bar
            close.append(price)
        df = pd.DataFrame({"close": close})
        rsi = RSI(period=period)
        result = rsi.calculate(df)
        valid = result[rsi.name].dropna()
        assert len(valid) > 0, "Should have valid RSI rows."
        last_valid = valid.iloc[-1]
        assert last_valid > 80, f"Expected RSI > 80 on mostly-rising prices, got {last_valid:.1f}"

    def test_all_losses_rsi_near_0(self) -> None:
        """Mostly falling prices (small rallies) → RSI should be very low (<20).
        Note: pure loss series give avg_gain=0 → RSI is NaN, so we use mostly-down prices."""
        period = 5
        close = []
        price = 200.0
        for i in range(60):
            price -= 1.0 if i % 10 != 9 else -0.05  # mostly down, tiny rally every 10th bar
            close.append(price)
        df = pd.DataFrame({"close": close})
        rsi = RSI(period=period)
        result = rsi.calculate(df)
        valid = result[rsi.name].dropna()
        assert len(valid) > 0, "Should have valid RSI rows."
        last_valid = valid.iloc[-1]
        assert last_valid < 20, f"Expected RSI < 20 on mostly-falling prices, got {last_valid:.1f}"


# ---------------------------------------------------------------------------
# Stochastic tests
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_k_and_d_columns_added(self) -> None:
        df = _make_df()
        stoch = Stochastic()
        result = stoch.calculate(df)
        assert stoch.name in result.columns
        assert stoch.name_d in result.columns

    def test_k_bounded_0_to_100(self) -> None:
        df = _make_df()
        stoch = Stochastic(k_period=14)
        result = stoch.calculate(df)
        valid = result[stoch.name].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_d_bounded_0_to_100(self) -> None:
        df = _make_df()
        stoch = Stochastic(k_period=14, d_period=3)
        result = stoch.calculate(df)
        valid = result[stoch.name_d].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            Stochastic().calculate(df)

    def test_column_names_reflect_periods(self) -> None:
        stoch = Stochastic(k_period=9, d_period=3)
        assert stoch.name == "stoch_k_9"
        assert stoch.name_d == "stoch_d_9_3"


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------

class TestMACD:
    def test_three_columns_added(self) -> None:
        df = _make_df()
        macd = MACD()
        result = macd.calculate(df)
        assert macd.name in result.columns
        assert macd.name_signal in result.columns
        assert macd.name_hist in result.columns

    def test_histogram_is_macd_minus_signal(self) -> None:
        """Histogram must equal MACD line minus signal line exactly."""
        df = _make_df()
        macd = MACD()
        result = macd.calculate(df)
        diff = result[macd.name] - result[macd.name_signal]
        pd.testing.assert_series_equal(result[macd.name_hist], diff, check_names=False)

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="close"):
            MACD().calculate(df)

    def test_column_names_reflect_periods(self) -> None:
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        assert macd.name == "macd_12_26"
        assert macd.name_signal == "macd_signal_12_26_9"
        assert macd.name_hist == "macd_hist_12_26_9"

    def test_valid_rows_exist(self) -> None:
        df = _make_df(length=100)
        macd = MACD()
        result = macd.calculate(df)
        assert result[macd.name].notna().any(), "MACD should produce some valid values."


# ---------------------------------------------------------------------------
# CCI tests
# ---------------------------------------------------------------------------

class TestCCI:
    def test_column_added(self) -> None:
        df = _make_df()
        cci = CCI(period=20)
        result = cci.calculate(df)
        assert cci.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert CCI(period=14).name == "cci_14"

    def test_warmup_period(self) -> None:
        period = 20
        df = _make_df()
        cci = CCI(period=period)
        result = cci.calculate(df)
        assert result[cci.name].iloc[: period - 1].isna().all()
        assert not np.isnan(result[cci.name].iloc[period - 1])

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            CCI().calculate(df)

    def test_produces_numeric_values(self) -> None:
        df = _make_df()
        cci = CCI(period=20)
        result = cci.calculate(df)
        valid = result[cci.name].dropna()
        assert valid.dtype.kind == "f"
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# ROC tests
# ---------------------------------------------------------------------------

class TestROC:
    def test_column_added(self) -> None:
        df = _make_df()
        roc = ROC(period=10)
        result = roc.calculate(df)
        assert roc.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert ROC(period=5).name == "roc_5"

    def test_warmup_period(self) -> None:
        period = 10
        df = _make_df()
        roc = ROC(period=period)
        result = roc.calculate(df)
        # pct_change(periods=N) → first N rows are NaN
        assert result[roc.name].iloc[:period].isna().all()

    def test_positive_when_price_rising(self) -> None:
        """Strictly increasing prices → ROC > 0 after warmup."""
        period = 5
        close = [float(i) for i in range(1, 31)]
        df = pd.DataFrame({"close": close})
        roc = ROC(period=period)
        result = roc.calculate(df)
        valid = result[roc.name].dropna()
        assert (valid > 0).all(), "ROC must be positive for monotonically rising prices."

    def test_negative_when_price_falling(self) -> None:
        """Strictly decreasing prices → ROC < 0 after warmup."""
        period = 5
        close = [float(30 - i) for i in range(30)]
        df = pd.DataFrame({"close": close})
        roc = ROC(period=period)
        result = roc.calculate(df)
        valid = result[roc.name].dropna()
        assert (valid < 0).all(), "ROC must be negative for monotonically falling prices."

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="close"):
            ROC().calculate(df)
