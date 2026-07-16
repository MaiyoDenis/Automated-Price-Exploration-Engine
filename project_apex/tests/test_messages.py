"""
Test Messages
"""

import pytest

from project_apex.api.messages import MessageBuilder, MessageParser
from project_apex.exceptions import DerivAPIError
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


@pytest.fixture
def builder():
    return MessageBuilder()


@pytest.fixture
def parser():
    return MessageParser()


def test_subscribe_ticks_structure(builder: MessageBuilder):
    req = builder.subscribe_ticks("R_25")
    assert "ticks" in req
    assert req["ticks"] == "R_25"
    assert req["subscribe"] == 1
    assert "req_id" in req


def test_authorize_structure(builder: MessageBuilder):
    req = builder.authorize("dummy_token")
    assert "authorize" in req
    assert req["authorize"] == "dummy_token"
    assert "req_id" in req


def test_ping_structure(builder: MessageBuilder):
    req = builder.ping()
    assert "ping" in req
    assert req["ping"] == 1
    assert "req_id" in req


def test_req_id_increments(builder: MessageBuilder):
    req1 = builder.ping()
    req2 = builder.ping()
    assert req1["req_id"] != req2["req_id"]


def test_empty_symbol_raises_value_error(builder: MessageBuilder):
    with pytest.raises(ValueError):
        builder.subscribe_ticks("")


def test_none_symbol_raises_value_error(builder: MessageBuilder):
    with pytest.raises(ValueError):
        builder.subscribe_ticks(None)  # type: ignore


def test_parse_tick_response(parser: MessageParser):
    raw = {
        "msg_type": "tick",
        "tick": {
            "symbol": "R_25",
            "epoch": 1000,
            "quote": 1.5
        }
    }
    parsed = parser.parse(raw)
    assert isinstance(parsed, Tick)
    assert parsed.symbol == "R_25"
    assert parsed.timestamp == 1000000
    assert parsed.price == 1.5


def test_parse_candles_response(parser: MessageParser):
    raw = {
        "msg_type": "candles",
        "echo_req": {"ticks_history": "R_25", "granularity": 60},
        "candles": [
            {"epoch": 1000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}
        ]
    }
    parsed = parser.parse(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    candle = parsed[0]
    assert isinstance(candle, Candle)
    assert candle.symbol == "R_25"
    assert candle.timeframe == 60
    assert candle.timestamp == 1000000
    assert candle.open == 1.0
    assert candle.high == 2.0
    assert candle.low == 0.5
    assert candle.close == 1.5


def test_parse_history_response(parser: MessageParser):
    raw = {
        "msg_type": "history",
        "echo_req": {"ticks_history": "R_25"},
        "history": {
            "prices": [1.5],
            "times": [1000]
        }
    }
    parsed = parser.parse(raw)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    tick = parsed[0]
    assert isinstance(tick, Tick)
    assert tick.symbol == "R_25"
    assert tick.timestamp == 1000000
    assert tick.price == 1.5


def test_parse_error_raises_deriv_api_error(parser: MessageParser):
    raw = {
        "error": {
            "code": "InvalidToken",
            "message": "Invalid API token."
        }
    }
    with pytest.raises(DerivAPIError) as exc_info:
        parser.parse(raw)
    assert exc_info.value.code == "InvalidToken"


def test_parse_unknown_msg_type_returns_none(parser: MessageParser):
    raw = {"msg_type": "unknown"}
    assert parser.parse(raw) is None


def test_round_trip_tick(parser: MessageParser):
    # Simulated roundtrip since there is no native serializer to raw API format
    tick = Tick("R_25", 1000000, 1.5)
    raw = {
        "msg_type": "tick",
        "tick": {
            "symbol": tick.symbol,
            "epoch": tick.timestamp // 1000,
            "quote": tick.price
        }
    }
    parsed = parser.parse(raw)
    assert tick == parsed


def test_round_trip_candle(parser: MessageParser):
    candle = Candle("R_25", 60, 1000000, 1.0, 2.0, 0.5, 1.5, 0)
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
    parsed = parser.parse(raw)
    assert isinstance(parsed, list)
    assert candle == parsed[0]
