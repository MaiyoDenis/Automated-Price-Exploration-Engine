"""
Test Validator
"""

from project_apex.data.validator import validate_tick, validate_candle
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


def test_valid_tick_passes():
    tick = Tick("R_25", 1000, 1.5)
    res = validate_tick(tick, None)
    assert res.is_valid is True


def test_empty_symbol_rejected():
    tick = Tick("", 1000, 1.5)
    res = validate_tick(tick, None)
    assert res.is_valid is False
    assert res.rule == "symbol_present"


def test_zero_price_rejected():
    tick = Tick("R_25", 1000, 0.0)
    res = validate_tick(tick, None)
    assert res.is_valid is False
    assert res.rule == "price_positive"


def test_negative_price_rejected():
    tick = Tick("R_25", 1000, -1.0)
    res = validate_tick(tick, None)
    assert res.is_valid is False
    assert res.rule == "price_positive"


def test_zero_timestamp_rejected():
    tick = Tick("R_25", 0, 1.5)
    res = validate_tick(tick, None)
    assert res.is_valid is False
    assert res.rule == "timestamp_positive"


def test_out_of_sequence_tick_rejected():
    tick = Tick("R_25", 1000, 1.5)
    res = validate_tick(tick, 2000)
    assert res.is_valid is False
    assert res.rule == "sequence"


def test_equal_timestamp_tick_accepted():
    tick = Tick("R_25", 1000, 1.5)
    res = validate_tick(tick, 1000)
    assert res.is_valid is True


def test_valid_candle_passes():
    candle = Candle("R_25", 60, 60000, 1.0, 2.0, 0.5, 1.5, 10)
    res = validate_candle(candle, [60, 300])
    assert res.is_valid is True


def test_candle_invalid_ohlc_low_gt_high():
    candle = Candle("R_25", 60, 60000, 1.0, 0.5, 2.0, 1.5, 10)
    res = validate_candle(candle, [60])
    assert res.is_valid is False
    assert res.rule == "ohlc_integrity"


def test_candle_open_below_low():
    candle = Candle("R_25", 60, 60000, 0.1, 2.0, 0.5, 1.5, 10)
    res = validate_candle(candle, [60])
    assert res.is_valid is False
    assert res.rule == "ohlc_integrity"


def test_candle_close_above_high():
    candle = Candle("R_25", 60, 60000, 1.0, 2.0, 0.5, 2.5, 10)
    res = validate_candle(candle, [60])
    assert res.is_valid is False
    assert res.rule == "ohlc_integrity"


def test_candle_invalid_timeframe():
    candle = Candle("R_25", 120, 120000, 1.0, 2.0, 0.5, 1.5, 10)
    res = validate_candle(candle, [60, 300])
    assert res.is_valid is False
    assert res.rule == "valid_timeframe"
