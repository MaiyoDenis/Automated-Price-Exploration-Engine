"""
Project APEX
Domain Models: Candle

This module defines the immutable Candle domain object.

Floor-Alignment Invariant:
The `timestamp` attribute of a Candle must always represent the start of the 
timeframe period, floor-aligned to the `timeframe` boundary. For example, 
if the timeframe is 60 seconds (1 minute), the timestamp must be a multiple 
of 60,000 milliseconds.

Formula: `timestamp = (tick.timestamp // (timeframe * 1000)) * (timeframe * 1000)`

This ensures that two independently built candles for the same symbol, 
timeframe, and period are identical, enabling idempotent database inserts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    """An OHLC aggregate for one instrument over a fixed timeframe interval.

    Attributes:
        symbol: The instrument identifier (e.g., "R_25").
        timeframe: The duration of the candle in seconds.
        timestamp: Floor-aligned Unix epoch timestamp in milliseconds.
        open: The opening price.
        high: The highest price during the period.
        low: The lowest price during the period.
        close: The closing price.
        tick_count: The number of ticks that formed this candle.
    """

    symbol: str
    timeframe: int
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    tick_count: int
