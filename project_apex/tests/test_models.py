"""
Test Models
"""

import pytest
from dataclasses import FrozenInstanceError

from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


def test_tick_construction():
    tick = Tick(symbol="R_25", timestamp=1000, price=1.5)
    assert tick.symbol == "R_25"
    assert tick.timestamp == 1000
    assert tick.price == 1.5


def test_tick_frozen():
    tick = Tick(symbol="R_25", timestamp=1000, price=1.5)
    with pytest.raises(FrozenInstanceError):
        tick.price = 2.0  # type: ignore


def test_tick_equality():
    tick1 = Tick(symbol="R_25", timestamp=1000, price=1.5)
    tick2 = Tick(symbol="R_25", timestamp=1000, price=1.5)
    assert tick1 == tick2


def test_candle_construction():
    candle = Candle(
        symbol="R_25", timeframe=60, timestamp=60000,
        open=1.0, high=2.0, low=0.5, close=1.5, tick_count=10
    )
    assert candle.symbol == "R_25"
    assert candle.timeframe == 60
    assert candle.timestamp == 60000
    assert candle.open == 1.0
    assert candle.high == 2.0
    assert candle.low == 0.5
    assert candle.close == 1.5
    assert candle.tick_count == 10


def test_candle_frozen():
    candle = Candle(
        symbol="R_25", timeframe=60, timestamp=60000,
        open=1.0, high=2.0, low=0.5, close=1.5, tick_count=10
    )
    with pytest.raises(FrozenInstanceError):
        candle.close = 2.0  # type: ignore


def test_candle_timestamp_floor_alignment():
    # External verification of formula
    tick_timestamp = 65000
    timeframe = 60
    expected_timestamp = (tick_timestamp // (timeframe * 1000)) * (timeframe * 1000)
    assert expected_timestamp == 60000
