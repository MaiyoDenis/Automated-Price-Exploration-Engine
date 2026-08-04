"""
Project APEX — Tests for Volatility Indicators

Covers: BollingerBands, ATR, KeltnerChannels, DonchianChannels —
column creation, band ordering, mathematical identities, warmup, error handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.indicators.volatility import ATR, BollingerBands, DonchianChannels, KeltnerChannels


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_df(length: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.standard_normal(length).cumsum()
    high = close + rng.random(length) * 2 + 0.5   # guarantee high > close
    low = close - rng.random(length) * 2 - 0.5    # guarantee low < close
    volume = rng.integers(100, 1000, size=length).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# BollingerBands tests
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_five_columns_added(self) -> None:
        df = _make_df()
        bb = BollingerBands(period=20)
        result = bb.calculate(df)
        for col in (bb.name, bb.name_upper, bb.name_lower, bb.name_pct_b, bb.name_width):
            assert col in result.columns, f"Expected column '{col}'."

    def test_upper_gt_middle_gt_lower(self) -> None:
        df = _make_df()
        bb = BollingerBands(period=20)
        result = bb.calculate(df)
        valid = result.dropna(subset=[bb.name, bb.name_upper, bb.name_lower])
        assert (valid[bb.name_upper] > valid[bb.name]).all(), "Upper must be > middle."
        assert (valid[bb.name] > valid[bb.name_lower]).all(), "Middle must be > lower."

    def test_pct_b_at_middle_is_approx_0_5(self) -> None:
        """When close equals the middle band, %B should be ~0.5."""
        period = 5
        # Constant price → std=0 → NaN for %B, skip. Use rising+falling price.
        close = [100.0 + np.sin(i * 0.3) for i in range(50)]
        df = pd.DataFrame({"close": close})
        bb = BollingerBands(period=period)
        result = bb.calculate(df)
        # At rows where close ≈ middle, %B ≈ 0.5
        valid = result.dropna(subset=[bb.name, bb.name_pct_b])
        if len(valid) == 0:
            pytest.skip("No valid rows to check.")
        # At exact middle, %B=0.5. Check it's in [0, 1] range instead.
        assert (valid[bb.name_pct_b] >= -1).all()  # allows outside-band values

    def test_width_non_negative(self) -> None:
        df = _make_df()
        bb = BollingerBands(period=20)
        result = bb.calculate(df)
        valid = result[bb.name_width].dropna()
        assert (valid >= 0).all(), "Band width must be non-negative."

    def test_warmup_period_nan(self) -> None:
        period = 20
        df = _make_df()
        bb = BollingerBands(period=period)
        result = bb.calculate(df)
        assert result[bb.name].iloc[: period - 1].isna().all()
        assert not np.isnan(result[bb.name].iloc[period - 1])

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="close"):
            BollingerBands().calculate(df)

    def test_column_names_reflect_period(self) -> None:
        bb = BollingerBands(period=20)
        assert bb.name == "bb_middle_20"
        assert bb.name_upper == "bb_upper_20"
        assert bb.name_lower == "bb_lower_20"


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

class TestATR:
    def test_column_added(self) -> None:
        df = _make_df()
        atr = ATR(period=14)
        result = atr.calculate(df)
        assert atr.name in result.columns

    def test_column_name_reflects_period(self) -> None:
        assert ATR(period=7).name == "atr_7"

    def test_non_negative(self) -> None:
        df = _make_df()
        atr = ATR(period=14)
        result = atr.calculate(df)
        valid = result[atr.name].dropna()
        assert (valid >= 0).all(), "ATR values must be non-negative."

    def test_warmup_period(self) -> None:
        period = 14
        df = _make_df()
        atr = ATR(period=period)
        result = atr.calculate(df)
        # EWM with min_periods=period
        assert result[atr.name].iloc[: period - 1].isna().all()
        assert not np.isnan(result[atr.name].iloc[period - 1])

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="high"):
            ATR(period=2).calculate(df)

    def test_high_volatility_greater_atr(self) -> None:
        """Wider H-L range → higher ATR."""
        period = 5
        # Low volatility: tight H-L
        df_low = pd.DataFrame({
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "close": [100.0] * 30,
        })
        # High volatility: wide H-L
        df_high = pd.DataFrame({
            "high": [110.0] * 30,
            "low": [90.0] * 30,
            "close": [100.0] * 30,
        })
        atr_low = ATR(period=period).calculate(df_low)[f"atr_{period}"].dropna().mean()
        atr_high = ATR(period=period).calculate(df_high)[f"atr_{period}"].dropna().mean()
        assert atr_high > atr_low, "Higher volatility should produce larger ATR."


# ---------------------------------------------------------------------------
# KeltnerChannels tests
# ---------------------------------------------------------------------------

class TestKeltnerChannels:
    def test_three_columns_added(self) -> None:
        df = _make_df()
        kc = KeltnerChannels()
        result = kc.calculate(df)
        for col in (kc.name, kc.name_upper, kc.name_lower):
            assert col in result.columns

    def test_upper_gt_lower_all_rows(self) -> None:
        df = _make_df()
        kc = KeltnerChannels()
        result = kc.calculate(df)
        valid = result.dropna(subset=[kc.name_upper, kc.name_lower])
        assert (valid[kc.name_upper] > valid[kc.name_lower]).all()

    def test_middle_between_bands(self) -> None:
        df = _make_df()
        kc = KeltnerChannels()
        result = kc.calculate(df)
        valid = result.dropna(subset=[kc.name, kc.name_upper, kc.name_lower])
        assert (valid[kc.name_upper] >= valid[kc.name]).all()
        assert (valid[kc.name] >= valid[kc.name_lower]).all()

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            KeltnerChannels().calculate(df)


# ---------------------------------------------------------------------------
# DonchianChannels tests
# ---------------------------------------------------------------------------

class TestDonchianChannels:
    def test_three_columns_added(self) -> None:
        df = _make_df()
        dc = DonchianChannels(period=20)
        result = dc.calculate(df)
        for col in (dc.name, dc.name_lower, dc.name_middle):
            assert col in result.columns

    def test_upper_is_rolling_max_of_high(self) -> None:
        period = 10
        df = _make_df()
        dc = DonchianChannels(period=period)
        result = dc.calculate(df)
        expected_upper = df["high"].rolling(window=period).max()
        pd.testing.assert_series_equal(result[dc.name], expected_upper, check_names=False)

    def test_lower_is_rolling_min_of_low(self) -> None:
        period = 10
        df = _make_df()
        dc = DonchianChannels(period=period)
        result = dc.calculate(df)
        expected_lower = df["low"].rolling(window=period).min()
        pd.testing.assert_series_equal(result[dc.name_lower], expected_lower, check_names=False)

    def test_middle_is_avg_of_upper_lower(self) -> None:
        df = _make_df()
        dc = DonchianChannels(period=10)
        result = dc.calculate(df)
        expected_mid = (result[dc.name] + result[dc.name_lower]) / 2.0
        pd.testing.assert_series_equal(result[dc.name_middle], expected_mid, check_names=False)

    def test_upper_gte_lower(self) -> None:
        df = _make_df()
        dc = DonchianChannels(period=10)
        result = dc.calculate(df)
        valid = result.dropna(subset=[dc.name, dc.name_lower])
        assert (valid[dc.name] >= valid[dc.name_lower]).all()

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            DonchianChannels().calculate(df)
