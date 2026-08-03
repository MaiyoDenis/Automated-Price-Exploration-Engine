"""
Tests for Technical Indicators.
"""
import numpy as np
import pandas as pd
from project_apex.indicators.oscillators import RSI, MACD
from project_apex.indicators.volatility import BollingerBands, ATR
from project_apex.indicators.trend import ADX


def _make_df(length: int = 100) -> pd.DataFrame:
    """Helper to generate a mock OHLCV dataframe."""
    np.random.seed(42)
    # Random walk
    close = 100 + np.random.randn(length).cumsum()
    high = close + np.random.rand(length) * 2
    low = close - np.random.rand(length) * 2
    open_p = close - np.random.randn(length)
    volume = np.random.randint(100, 1000, size=length)
    
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


def test_rsi():
    df = _make_df()
    rsi = RSI(period=14)
    res = rsi.calculate(df)
    assert rsi.name in res.columns
    # First 14 values should be NaN
    assert np.isnan(res[rsi.name].iloc[13])
    assert not np.isnan(res[rsi.name].iloc[14])
    # RSI bounded 0-100
    assert res[rsi.name].max() <= 100
    assert res[rsi.name].min() >= 0


def test_macd():
    df = _make_df()
    macd = MACD(fast_period=12, slow_period=26, signal_period=9)
    res = macd.calculate(df)
    assert macd.name in res.columns
    assert macd.name_signal in res.columns
    assert macd.name_hist in res.columns
    
    valid = res.dropna()
    assert len(valid) > 0


def test_bollinger_bands():
    df = _make_df()
    bb = BollingerBands(period=20, num_std=2.0)
    res = bb.calculate(df)
    assert bb.name_upper in res.columns
    assert bb.name_lower in res.columns
    assert bb.name_width in res.columns
    assert bb.name_pct_b in res.columns
    
    # Check width > 0
    valid = res.dropna()
    assert (valid[bb.name_width] >= 0).all()


def test_atr():
    df = _make_df()
    atr = ATR(period=14)
    res = atr.calculate(df)
    assert atr.name in res.columns
    valid = res.dropna()
    assert (valid[atr.name] >= 0).all()


def test_adx():
    df = _make_df()
    adx = ADX(period=14)
    res = adx.calculate(df)
    assert adx.name in res.columns
    assert adx.name_plus_di in res.columns
    assert adx.name_minus_di in res.columns
    
    valid = res.dropna()
    assert (valid[adx.name] >= 0).all()
    assert (valid[adx.name] <= 100).all()
