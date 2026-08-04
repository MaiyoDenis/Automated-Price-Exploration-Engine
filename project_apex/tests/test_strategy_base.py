"""
Project APEX — Tests for Strategy Base Classes

Covers: Strategy (batch) and LiveStrategy (event-driven) base classes —
indicator composition, prepare_data, initialize hook, on_tick default,
candle history accumulation, and history DataFrame schema.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from project_apex.indicators.oscillators import RSI
from project_apex.indicators.moving_averages import SMA
from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy, Strategy
from project_apex.strategies.signals import SignalType, TradeSignal


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class ConcreteStrategy(Strategy):
    """Minimal concrete Strategy that just returns data unchanged."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["signal"] = 0
        return data


class ConcreteSignalStrategy(Strategy):
    """Returns signal=1 always after prepare_data."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_data(data)
        df["signal"] = 1
        return df


class TrackInitLiveStrategy(LiveStrategy):
    """Records whether initialize() was called and what config it received."""

    initialized_with: dict[str, Any] = {}
    init_called: bool = False

    def initialize(self, config: dict[str, Any]) -> None:
        TrackInitLiveStrategy.init_called = True
        TrackInitLiveStrategy.initialized_with = config

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        return None


class BuyOnEveryCandleLiveStrategy(LiveStrategy):
    """Emits a BUY signal on every candle."""

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        self._append_candle(candle)
        return TradeSignal(
            symbol=candle.symbol,
            signal_type=SignalType.BUY,
            confidence=1.0,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(length: int = 30) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal(length).cumsum()
    return pd.DataFrame({"close": close})


def _make_candle(symbol: str = "R_25", timestamp: int = 1_000_000, price: float = 100.0) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=60,
        timestamp=timestamp,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        tick_count=10,
    )


def _make_tick(symbol: str = "R_25", timestamp: int = 1_000_000, price: float = 100.0) -> Tick:
    return Tick(symbol=symbol, timestamp=timestamp, price=price)


# ===========================================================================
# Strategy (batch) tests
# ===========================================================================

class TestStrategy:
    def test_initial_indicators_empty(self) -> None:
        strategy = ConcreteStrategy("test")
        assert strategy.indicators == []

    def test_add_indicator_appended(self) -> None:
        strategy = ConcreteStrategy("test")
        rsi = RSI(period=14)
        strategy.add_indicator(rsi)
        assert len(strategy.indicators) == 1
        assert strategy.indicators[0] is rsi

    def test_add_multiple_indicators(self) -> None:
        strategy = ConcreteStrategy("test")
        strategy.add_indicator(SMA(period=5))
        strategy.add_indicator(RSI(period=14))
        assert len(strategy.indicators) == 2

    def test_prepare_data_applies_single_indicator(self) -> None:
        strategy = ConcreteStrategy("test")
        rsi = RSI(period=5)
        strategy.add_indicator(rsi)
        df = _make_df(length=30)
        result = strategy.prepare_data(df)
        assert rsi.name in result.columns, f"Expected '{rsi.name}' from prepare_data."

    def test_prepare_data_applies_all_indicators_in_order(self) -> None:
        strategy = ConcreteStrategy("test")
        sma = SMA(period=5)
        rsi = RSI(period=5)
        strategy.add_indicator(sma)
        strategy.add_indicator(rsi)
        df = _make_df(length=30)
        result = strategy.prepare_data(df)
        assert sma.name in result.columns
        assert rsi.name in result.columns

    def test_prepare_data_does_not_mutate_original(self) -> None:
        """prepare_data works on a copy — the original df should NOT gain indicator columns."""
        strategy = ConcreteStrategy("test")
        strategy.add_indicator(SMA(period=5))
        original = _make_df(length=30)
        original_cols = set(original.columns)
        strategy.prepare_data(original)
        # prepare_data does df = data.copy() so the caller's df is not modified
        assert set(original.columns) == original_cols, (
            "prepare_data must not mutate the caller's DataFrame."
        )

    def test_generate_signals_adds_signal_column(self) -> None:
        strategy = ConcreteStrategy("test")
        df = _make_df()
        result = strategy.generate_signals(df)
        assert "signal" in result.columns

    def test_name_stored_correctly(self) -> None:
        strategy = ConcreteStrategy("my_strategy")
        assert strategy.name == "my_strategy"


# ===========================================================================
# LiveStrategy tests
# ===========================================================================

class TestLiveStrategy:
    def setup_method(self) -> None:
        # Reset class-level state before each test
        TrackInitLiveStrategy.init_called = False
        TrackInitLiveStrategy.initialized_with = {}

    def test_initialize_called_at_construction(self) -> None:
        config = {"rsi_period": 14, "threshold": 70}
        TrackInitLiveStrategy("tracker", config=config)
        assert TrackInitLiveStrategy.init_called is True

    def test_initialize_receives_correct_config(self) -> None:
        config = {"rsi_period": 14, "threshold": 70}
        TrackInitLiveStrategy("tracker", config=config)
        assert TrackInitLiveStrategy.initialized_with == config

    def test_initialize_with_no_config(self) -> None:
        TrackInitLiveStrategy("tracker")
        assert TrackInitLiveStrategy.init_called is True
        assert TrackInitLiveStrategy.initialized_with == {}

    def test_on_tick_returns_none_by_default(self) -> None:
        strategy = TrackInitLiveStrategy("tracker")
        tick = _make_tick()
        assert strategy.on_tick(tick) is None

    def test_name_stored_correctly(self) -> None:
        strategy = TrackInitLiveStrategy("live_strat")
        assert strategy.name == "live_strat"

    def test_config_stored_correctly(self) -> None:
        config = {"key": "value"}
        strategy = TrackInitLiveStrategy("s", config=config)
        assert strategy.config == config

    def test_append_candle_grows_history(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        c1 = _make_candle(timestamp=1_000_000)
        c2 = _make_candle(timestamp=1_000_060)
        strategy._append_candle(c1)
        history = strategy._append_candle(c2)
        assert len(history) == 2

    def test_append_candle_stores_per_symbol_timeframe(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        c_r25 = _make_candle(symbol="R_25", timestamp=1_000_000)
        c_r50 = Candle(
            symbol="R_50", timeframe=60, timestamp=1_000_000,
            open=200.0, high=201.0, low=199.0, close=200.0, tick_count=5
        )
        strategy._append_candle(c_r25)
        strategy._append_candle(c_r50)
        # Each symbol has separate history
        assert len(strategy._candle_history[("R_25", 60)]) == 1
        assert len(strategy._candle_history[("R_50", 60)]) == 1

    def test_get_history_df_columns(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        candle = _make_candle()
        strategy._append_candle(candle)
        df = strategy._get_history_df(candle)
        expected_cols = {"timestamp", "open", "high", "low", "close", "tick_count"}
        assert expected_cols.issubset(set(df.columns))

    def test_get_history_df_empty_when_no_candles(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        candle = _make_candle()
        df = strategy._get_history_df(candle)
        assert df.empty

    def test_get_history_df_row_count_matches_appended(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        candles = [_make_candle(timestamp=1_000_000 + i * 60) for i in range(5)]
        for c in candles:
            strategy._append_candle(c)
        df = strategy._get_history_df(candles[0])
        assert len(df) == 5

    def test_on_candle_emits_trade_signal(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        candle = _make_candle()
        signal = strategy.on_candle(candle)
        assert signal is not None
        assert isinstance(signal, TradeSignal)
        assert signal.signal_type == SignalType.BUY

    def test_on_candle_signal_price_matches_candle_close(self) -> None:
        strategy = BuyOnEveryCandleLiveStrategy("buyer")
        candle = _make_candle(price=123.45)
        signal = strategy.on_candle(candle)
        assert signal is not None
        assert signal.price == pytest.approx(123.45)
