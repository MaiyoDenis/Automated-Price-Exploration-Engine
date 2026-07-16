"""
Project APEX
Candle Builder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from loguru import logger

from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


@dataclass
class CandleAccumulator:
    symbol: str
    timeframe: int
    period_start: int
    open: float
    high: float
    low: float
    close: float
    tick_count: int


class CandleBuilder:
    """Builds candles incrementally from a stream of ticks."""

    def __init__(self, timeframes: list[int], valid_timeframes: list[int]) -> None:
        self.timeframes = timeframes
        self.valid_timeframes = valid_timeframes
        self._accumulators: Dict[Tuple[str, int], CandleAccumulator] = {}

    def process_tick(self, tick: Tick) -> list[Candle]:
        """Synchronously processes a tick and returns any completed candles."""
        completed_candles: list[Candle] = []

        for tf in self.timeframes:
            if tf not in self.valid_timeframes:
                continue

            period_start = (tick.timestamp // (tf * 1000)) * (tf * 1000)
            key = (tick.symbol, tf)
            acc = self._accumulators.get(key)

            if acc is None:
                self._accumulators[key] = CandleAccumulator(
                    symbol=tick.symbol,
                    timeframe=tf,
                    period_start=period_start,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    tick_count=1
                )
            elif period_start == acc.period_start:
                acc.high = max(acc.high, tick.price)
                acc.low = min(acc.low, tick.price)
                acc.close = tick.price
                acc.tick_count += 1
            else:
                # Boundary crossed, finalize current accumulator
                completed_candles.append(
                    Candle(
                        symbol=acc.symbol,
                        timeframe=acc.timeframe,
                        timestamp=acc.period_start,
                        open=acc.open,
                        high=acc.high,
                        low=acc.low,
                        close=acc.close,
                        tick_count=acc.tick_count
                    )
                )
                # Open new accumulator
                self._accumulators[key] = CandleAccumulator(
                    symbol=tick.symbol,
                    timeframe=tf,
                    period_start=period_start,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    tick_count=1
                )

        return completed_candles

    def stop(self) -> None:
        """Discards in-progress accumulators and clears state."""
        in_progress_count = len(self._accumulators)
        if in_progress_count > 0:
            logger.warning(f"Discarding {in_progress_count} in-progress candle accumulators.")
        self._accumulators.clear()
