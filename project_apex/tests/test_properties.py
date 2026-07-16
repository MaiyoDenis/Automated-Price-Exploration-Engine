"""
Property-Based Tests
"""

import pytest
from hypothesis import given, strategies as st

from project_apex.models.tick import Tick
from project_apex.models.candle import Candle
from project_apex.api.messages import MessageBuilder, MessageParser
from project_apex.data.validator import validate_tick
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.database.sqlite_market_data import SQLiteMarketDataRepository
from project_apex.data.candle_builder import CandleBuilder


@pytest.fixture
def parser():
    return MessageParser()


# 1. Tick round-trip
@given(
    st.text(min_size=1),
    st.integers(min_value=1, max_value=2**31 - 1),
    st.floats(min_value=0.01, allow_nan=False, allow_infinity=False)
)
def test_tick_round_trip(symbol: str, timestamp: int, price: float):
    # In API, epoch is seconds, so timestamp (ms) must be a multiple of 1000 for perfect round-trip
    timestamp = (timestamp // 1000) * 1000
    if timestamp == 0:
        timestamp = 1000
        
    tick = Tick(symbol, timestamp, price)
    
    raw = {
        "msg_type": "tick",
        "tick": {
            "symbol": tick.symbol,
            "epoch": tick.timestamp // 1000,
            "quote": tick.price
        }
    }
    
    parser = MessageParser()
    parsed = parser.parse(raw)
    assert tick == parsed


# 2. Candle round-trip
@given(
    st.text(min_size=1),
    st.sampled_from([60, 300, 900]),
    st.integers(min_value=1, max_value=2**31 - 1),
    st.floats(min_value=0.01, max_value=1000.0),
    st.floats(min_value=0.01, max_value=1000.0),
    st.floats(min_value=0.01, max_value=1000.0),
    st.floats(min_value=0.01, max_value=1000.0)
)
def test_candle_round_trip(symbol: str, timeframe: int, timestamp: int, open_p: float, high: float, low: float, close: float):
    timestamp = (timestamp // 1000) * 1000
    if timestamp == 0:
        timestamp = 1000
        
    candle = Candle(symbol, timeframe, timestamp, open_p, high, low, close, 0)
    
    raw = {
        "msg_type": "candles",
        "echo_req": {"ticks_history": candle.symbol, "granularity": candle.timeframe},
        "candles": [
            {
                "epoch": candle.timestamp // 1000,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close
            }
        ]
    }
    
    parser = MessageParser()
    parsed = parser.parse(raw)
    assert candle == parsed[0]


# 3. Candle floor alignment
@given(
    st.integers(min_value=1, max_value=2**31 - 1),
    st.sampled_from([60, 300, 900])
)
def test_candle_floor_alignment(timestamp: int, timeframe: int):
    builder = CandleBuilder(timeframes=[timeframe], valid_timeframes=[timeframe])
    tick = Tick("R_25", timestamp, 1.5)
    
    builder.process_tick(tick)
    acc = builder._accumulators[("R_25", timeframe)]
    
    expected_start = (timestamp // (timeframe * 1000)) * (timeframe * 1000)
    assert acc.period_start == expected_start


# 4. Validation monotonicity
@given(
    st.lists(
        st.integers(min_value=1, max_value=100000), 
        min_size=1, 
        max_size=50
    ).map(sorted)
)
def test_validation_monotonicity(timestamps: list[int]):
    latest_timestamp = None
    for t in timestamps:
        tick = Tick("R_25", t, 1.5)
        res = validate_tick(tick, latest_timestamp)
        assert res.is_valid is True
        latest_timestamp = t


# 5. Duplicate insert idempotency
@given(
    st.text(min_size=1, max_size=10),
    st.integers(min_value=1, max_value=10000),
    st.floats(min_value=0.1, max_value=100.0)
)
def test_duplicate_insert_idempotency(symbol: str, timestamp: int, price: float):
    manager = SQLiteManager(":memory:")
    manager.connect()
    repo = SQLiteMarketDataRepository(manager)
    repo.initialize()
    
    tick = Tick(symbol, timestamp, price)
    repo.save_tick(tick)
    repo.save_tick(tick)  # Duplicate
    
    ticks = repo.get_ticks(symbol, timestamp, timestamp)
    assert len(ticks) == 1
    
    manager.close()


# 6. MessageBuilder req_id uniqueness
@given(st.integers(min_value=2, max_value=100))
def test_builder_req_id_uniqueness(n: int):
    builder = MessageBuilder()
    req_ids = set()
    for _ in range(n):
        req = builder.ping()
        req_ids.add(req["req_id"])
        
    assert len(req_ids) == n


# 7. Candle OHLC invariant
@given(
    st.lists(
        st.floats(min_value=0.1, max_value=100.0), 
        min_size=1, 
        max_size=50
    )
)
def test_candle_ohlc_invariant(prices: list[float]):
    builder = CandleBuilder([60], [60])
    
    for i, p in enumerate(prices):
        tick = Tick("R_25", 1000 + i*10, p) # All within same minute
        builder.process_tick(tick)
        
    # Cross boundary to finalize candle
    builder.process_tick(Tick("R_25", 61000, 1.0))
    
    acc = builder._accumulators[("R_25", 60)]
    assert acc.low <= acc.open <= acc.high
    assert acc.low <= acc.close <= acc.high
