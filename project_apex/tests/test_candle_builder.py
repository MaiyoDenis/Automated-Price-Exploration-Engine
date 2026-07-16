"""
Test Candle Builder
"""

import pytest
from loguru import logger

from project_apex.data.candle_builder import CandleBuilder
from project_apex.models.tick import Tick


@pytest.fixture
def log_sink():
    """Capture loguru messages into a list for assertion."""
    messages = []
    sink_id = logger.add(lambda msg: messages.append(msg), format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def builder():
    return CandleBuilder(timeframes=[60, 300], valid_timeframes=[60, 300])


def test_first_tick_opens_accumulator(builder: CandleBuilder):
    tick = Tick("R_25", 60000, 1.5)
    candles = builder.process_tick(tick)
    assert len(candles) == 0
    assert len(builder._accumulators) == 2


def test_tick_within_boundary_updates_ohlc(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_25", 65000, 2.0)
    tick3 = Tick("R_25", 66000, 1.0)
    
    builder.process_tick(tick1)
    builder.process_tick(tick2)
    builder.process_tick(tick3)
    
    acc_60 = builder._accumulators[("R_25", 60)]
    assert acc_60.high == 2.0
    assert acc_60.low == 1.0
    assert acc_60.close == 1.0
    assert acc_60.tick_count == 3


def test_tick_open_field_unchanged(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_25", 65000, 2.0)
    
    builder.process_tick(tick1)
    builder.process_tick(tick2)
    
    acc_60 = builder._accumulators[("R_25", 60)]
    assert acc_60.open == 1.5


def test_boundary_crossing_emits_candle(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_25", 125000, 2.0)
    
    builder.process_tick(tick1)
    candles = builder.process_tick(tick2)
    
    assert len(candles) == 1
    candle = candles[0]
    assert candle.timeframe == 60
    assert candle.timestamp == 60000


def test_boundary_crossing_opens_new_accumulator(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_25", 125000, 2.0)
    
    builder.process_tick(tick1)
    builder.process_tick(tick2)
    
    acc_60 = builder._accumulators[("R_25", 60)]
    assert acc_60.period_start == 120000
    assert acc_60.open == 2.0


def test_multi_timeframe_independence(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_25", 125000, 2.0)
    
    builder.process_tick(tick1)
    candles = builder.process_tick(tick2)
    
    # 60s crossed boundary, 300s did not
    assert len(candles) == 1
    assert candles[0].timeframe == 60
    
    acc_300 = builder._accumulators[("R_25", 300)]
    assert acc_300.tick_count == 2
    assert acc_300.period_start == 0


def test_multi_symbol_independence(builder: CandleBuilder):
    tick1 = Tick("R_25", 60000, 1.5)
    tick2 = Tick("R_100", 60000, 2.0)
    
    builder.process_tick(tick1)
    builder.process_tick(tick2)
    
    assert len(builder._accumulators) == 4  # 2 timeframes * 2 symbols


def test_stop_discards_in_progress(builder: CandleBuilder, log_sink):
    tick = Tick("R_25", 60000, 1.5)
    builder.process_tick(tick)
    
    builder.stop()
    assert len(builder._accumulators) == 0
    # Verify the warning was emitted
    combined = " ".join(log_sink)
    assert "Discarding 2 in-progress candle accumulators." in combined
