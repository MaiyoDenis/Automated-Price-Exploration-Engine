"""
Project APEX
Deriv API Message Builder and Parser
"""

from __future__ import annotations

import itertools
from typing import Any

from loguru import logger

from project_apex.exceptions import DerivAPIError
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


# Field constants shared between builder and parser
FIELD_SYMBOL = "symbol"
FIELD_EPOCH = "epoch"
FIELD_QUOTE = "quote"
FIELD_MSG_TYPE = "msg_type"
FIELD_TICK = "tick"
FIELD_CANDLES = "candles"
FIELD_HISTORY = "history"
FIELD_PRICES = "prices"
FIELD_TIMES = "times"
FIELD_OPEN = "open"
FIELD_HIGH = "high"
FIELD_LOW = "low"
FIELD_CLOSE = "close"
FIELD_GRANULARITY = "granularity"
FIELD_REQ_ID = "req_id"
FIELD_ERROR = "error"
FIELD_CODE = "code"
FIELD_MESSAGE = "message"
FIELD_SUBSCRIPTION = "subscription"
FIELD_ID = "id"


class MessageBuilder:
    """Builds typed requests for the Deriv API."""

    def __init__(self) -> None:
        self._req_id_counter = itertools.count(start=1)

    def _next_req_id(self) -> int:
        return next(self._req_id_counter)

    def _validate_required_string(self, param_name: str, value: Any) -> None:
        if value is None or value == "":
            raise ValueError(f"Parameter '{param_name}' cannot be None or empty string.")

    def authorize(self, token: str) -> dict[str, Any]:
        self._validate_required_string("token", token)
        return {
            "authorize": token,
            FIELD_REQ_ID: self._next_req_id(),
        }

    def subscribe_ticks(self, symbol: str) -> dict[str, Any]:
        self._validate_required_string(FIELD_SYMBOL, symbol)
        return {
            "ticks": symbol,
            "subscribe": 1,
            FIELD_REQ_ID: self._next_req_id(),
        }

    def unsubscribe(self, subscription_id: str) -> dict[str, Any]:
        self._validate_required_string(FIELD_SUBSCRIPTION, subscription_id)
        return {
            "forget": subscription_id,
            FIELD_REQ_ID: self._next_req_id(),
        }

    def ping(self) -> dict[str, Any]:
        return {
            "ping": 1,
            FIELD_REQ_ID: self._next_req_id(),
        }

    def history_ticks(self, symbol: str, start: int, end: int) -> dict[str, Any]:
        self._validate_required_string(FIELD_SYMBOL, symbol)
        return {
            "ticks_history": symbol,
            "end": str(end // 1000) if end else "latest",  # assuming epoch ms, API expects seconds or 'latest'
            "start": start // 1000 if start else 1,
            "style": "ticks",
            FIELD_REQ_ID: self._next_req_id(),
        }

    def history_candles(self, symbol: str, granularity: int, start: int, end: int) -> dict[str, Any]:
        self._validate_required_string(FIELD_SYMBOL, symbol)
        return {
            "ticks_history": symbol,
            "end": str(end // 1000) if end else "latest",
            "start": start // 1000 if start else 1,
            "style": "candles",
            FIELD_GRANULARITY: granularity,
            FIELD_REQ_ID: self._next_req_id(),
        }


class MessageParser:
    """Parses raw JSON dicts from the Deriv API into domain models."""

    def parse(self, raw: dict[str, Any]) -> Tick | list[Tick] | list[Candle] | None:
        if FIELD_ERROR in raw:
            error_data = raw[FIELD_ERROR]
            code = error_data.get(FIELD_CODE, "UNKNOWN_ERROR")
            message = error_data.get(FIELD_MESSAGE, "Unknown error occurred.")
            raise DerivAPIError(code, message)

        msg_type = raw.get(FIELD_MSG_TYPE)

        if msg_type == "tick":
            tick_data = raw.get(FIELD_TICK, {})
            return Tick(
                symbol=tick_data.get(FIELD_SYMBOL, ""),
                timestamp=int(tick_data.get(FIELD_EPOCH, 0)) * 1000,
                price=float(tick_data.get(FIELD_QUOTE, 0.0)),
            )

        if msg_type == "history":
            history_data = raw.get(FIELD_HISTORY, {})
            prices = history_data.get(FIELD_PRICES, [])
            times = history_data.get(FIELD_TIMES, [])
            symbol = raw.get("echo_req", {}).get("ticks_history", "")
            
            ticks = []
            for t, p in zip(times, prices):
                ticks.append(Tick(
                    symbol=symbol,
                    timestamp=int(t) * 1000,
                    price=float(p)
                ))
            return ticks

        if msg_type == "candles":
            candles_data = raw.get(FIELD_CANDLES, [])
            symbol = raw.get("echo_req", {}).get("ticks_history", "")
            timeframe = raw.get("echo_req", {}).get(FIELD_GRANULARITY, 60)
            
            candles = []
            for c in candles_data:
                candles.append(Candle(
                    symbol=symbol,
                    timeframe=int(timeframe),
                    timestamp=int(c.get(FIELD_EPOCH, 0)) * 1000,
                    open=float(c.get(FIELD_OPEN, 0.0)),
                    high=float(c.get(FIELD_HIGH, 0.0)),
                    low=float(c.get(FIELD_LOW, 0.0)),
                    close=float(c.get(FIELD_CLOSE, 0.0)),
                    tick_count=0, # History doesn't have tick_count
                ))
            return candles

        if msg_type in ("pong", "ping"):
            return None

        # Ignore authorize responses here since they don't produce a domain object
        if msg_type == "authorize":
            return None

        logger.warning(f"MessageParser encountered unknown msg_type: '{msg_type}'. Payload: {raw}")
        return None
