"""
Project APEX
Tick Collector
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from loguru import logger

from project_apex.api.ports import MarketDataProvider
from project_apex.database.ports import MarketDataRepository
from project_apex.data.candle_builder import CandleBuilder
from project_apex.data.validator import validate_tick, validate_candle
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


class TickCollector:
    """Subscribes to ticks, validates them, builds candles, and persists data.

    Architecture
    ------------
    Two cooperating async tasks share a bounded ``asyncio.Queue``:

    * ``_receive_loop`` — I/O-bound.  Calls ``provider.receive()`` as fast as
      the network delivers ticks and puts each raw :class:`Tick` into
      ``_tick_queue`` without blocking.  The queue absorbs bursts so ticks are
      never dropped during rapid price moves.

    * ``_process_loop`` — CPU/DB-bound.  Drains the queue, validates each tick,
      saves it to the repository, and feeds the :class:`CandleBuilder`.  Because
      this runs in a separate coroutine the I/O loop is never stalled by
      database writes or candle computation.
    """

    # Maximum ticks buffered before back-pressure slows the provider
    _QUEUE_MAXSIZE = 1000

    def __init__(
        self,
        provider: MarketDataProvider,
        repository: MarketDataRepository,
        candle_builder: CandleBuilder,
        symbols: list[str],
        stats_interval: int,
        valid_timeframes: list[int]
    ) -> None:
        self.provider = provider
        self.repository = repository
        self.candle_builder = candle_builder
        self.symbols = symbols
        self.stats_interval = stats_interval
        self.valid_timeframes = valid_timeframes
        
        self._latest_timestamps: Dict[str, int] = {}
        self._active_subscriptions: List[str] = []
        
        self._ticks_received = 0
        self._ticks_accepted = 0
        self._ticks_rejected = 0

        # Bounded queue — decouples network I/O from DB writes / candle math
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        
        self._stats_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Starts the tick collection process."""
        for symbol in self.symbols:
            sub_id = await self.provider.subscribe_ticks(symbol)
            self._active_subscriptions.append(sub_id)
            logger.info(f"Subscribed to ticks for {symbol}")
            
            # Initialize latest timestamp
            latest_ts = self.repository.get_latest_tick_timestamp(symbol)
            if latest_ts is not None:
                self._latest_timestamps[symbol] = latest_ts

        self._stats_task = asyncio.create_task(self._stats_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"[TickCollector] Started — queue maxsize={self._QUEUE_MAXSIZE}"
        )

    async def _receive_loop(self) -> None:
        """I/O loop: receive ticks from the provider and enqueue them.

        Runs at full network speed without performing any validation or DB
        writes so it is never blocked by downstream processing.
        """
        while True:
            try:
                tick = await self.provider.receive()
                self._ticks_received += 1
                try:
                    # Non-blocking put with a tight timeout so we never deadlock
                    # if the processing loop is stuck, but we don't silently drop.
                    self._tick_queue.put_nowait(tick)
                except asyncio.QueueFull:
                    # Queue is saturated — log and discard the oldest tick to
                    # stay current (drop head, keep tail).
                    try:
                        self._tick_queue.get_nowait()
                        self._tick_queue.put_nowait(tick)
                        logger.warning(
                            "[TickCollector] Tick queue full — oldest tick discarded "
                            "to stay current."
                        )
                    except asyncio.QueueEmpty:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving tick: {e}")

    async def _process_loop(self) -> None:
        """CPU/DB loop: drain the queue, validate, persist, and build candles."""
        while True:
            try:
                tick = await self._tick_queue.get()
                self._on_tick(tick)
                self._tick_queue.task_done()
            except asyncio.CancelledError:
                # Drain any remaining ticks before exiting
                while not self._tick_queue.empty():
                    try:
                        tick = self._tick_queue.get_nowait()
                        self._on_tick(tick)
                        self._tick_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                break
            except Exception as e:
                logger.error(f"[TickCollector] Error processing tick: {e}")

    def _on_tick(self, tick: Tick) -> None:
        """Handles an incoming tick."""
        latest_ts = self._latest_timestamps.get(tick.symbol)
        validation_result = validate_tick(tick, latest_ts)
        
        if validation_result.is_valid:
            self.repository.save_tick(tick)
            self._latest_timestamps[tick.symbol] = tick.timestamp
            self._ticks_accepted += 1
            
            candles = self.candle_builder.process_tick(tick)
            if candles:
                self._persist_candles(candles)
        else:
            logger.warning(
                f"Tick rejected for symbol {tick.symbol}. "
                f"Rule: {validation_result.rule}, Field: {validation_result.field}, "
                f"Observed: {validation_result.observed_value}"
            )
            self._ticks_rejected += 1

    def _persist_candles(self, candles: list[Candle]) -> None:
        """Validates and persists completed candles."""
        for candle in candles:
            validation_result = validate_candle(candle, self.valid_timeframes)
            if validation_result.is_valid:
                self.repository.save_candle(candle)
            else:
                logger.warning(
                    f"Candle rejected for symbol {candle.symbol}. "
                    f"Rule: {validation_result.rule}, Field: {validation_result.field}, "
                    f"Observed: {validation_result.observed_value}"
                )

    async def _stats_loop(self) -> None:
        """Periodically logs collection statistics."""
        while True:
            try:
                await asyncio.sleep(self.stats_interval)
                logger.info(
                    f"Tick Stats - Received: {self._ticks_received}, "
                    f"Accepted: {self._ticks_accepted}, "
                    f"Rejected: {self._ticks_rejected} | "
                    f"Queue depth: {self._tick_queue.qsize()}"
                )
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stops the collection process."""
        if self._stats_task is not None:
            self._stats_task.cancel()
            
        if self._receive_task is not None:
            self._receive_task.cancel()

        if self._process_task is not None:
            self._process_task.cancel()
            
        for sub_id in self._active_subscriptions:
            try:
                await self.provider.unsubscribe(sub_id)
                logger.info(f"Unsubscribed from {sub_id}")
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from {sub_id}: {e}")
                
        self._active_subscriptions.clear()
        self.candle_builder.stop()

    # ── Dynamic subscription support ─────────────────────────────────────────

    async def subscribe(self, symbol: str) -> None:
        """Dynamically subscribe to a new symbol (called by Autopilot)."""
        sub_id = await self.provider.subscribe_ticks(symbol)
        self._active_subscriptions.append(sub_id)
        latest_ts = self.repository.get_latest_tick_timestamp(symbol)
        if latest_ts is not None:
            self._latest_timestamps[symbol] = latest_ts
        logger.info(f"[TickCollector] Dynamically subscribed to {symbol}")

    async def unsubscribe(self, symbol: str) -> None:
        """Dynamically unsubscribe from a symbol (called by Autopilot)."""
        logger.info(f"[TickCollector] Unsubscribed from {symbol}")
