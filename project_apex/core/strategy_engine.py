"""
Project APEX — Strategy Engine

Dispatches completed candles to registered strategies and routes emitted signals
to the RiskEngine for approval. Tracks per-strategy performance.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Awaitable

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal


SignalHandler = Callable[[TradeSignal], Awaitable[None]]


class StrategyEngine:
    """
    Orchestrates multiple live strategies.

    - Receives candles from the ``TickCollector`` via :meth:`on_candle`.
    - Dispatches each candle to all registered strategies.
    - Collects any emitted :class:`TradeSignal` objects.
    - Calls registered signal handlers (e.g. RiskEngine) for each signal.

    Usage::

        engine = StrategyEngine()
        engine.register(my_strategy)
        engine.add_signal_handler(risk_engine.evaluate)
        # Then pass candles from the tick collector to engine.on_candle()
    """

    def __init__(self) -> None:
        self._strategies: list[LiveStrategy] = []
        self._signal_handlers: list[SignalHandler] = []

        # Performance tracking per strategy
        self._signal_counts: dict[str, int] = {}

    def register(self, strategy: LiveStrategy) -> None:
        """Register a live strategy to receive candle events."""
        self._strategies.append(strategy)
        self._signal_counts[strategy.name] = 0
        logger.info(f"[StrategyEngine] Registered strategy: {strategy.name}")

    def add_signal_handler(self, handler: SignalHandler) -> None:
        """
        Register an async handler that receives emitted TradeSignals.
        Handlers are called in registration order.
        """
        self._signal_handlers.append(handler)

    async def on_candle(self, candle: Candle) -> None:
        """
        Called by the tick pipeline for each completed candle.
        Dispatches to all strategies and routes signals to handlers.
        """
        for strategy in self._strategies:
            try:
                signal = strategy.on_candle(candle)
                if signal is not None:
                    self._signal_counts[strategy.name] += 1
                    await self._dispatch_signal(signal)
            except Exception as exc:
                logger.error(
                    f"[StrategyEngine] Error in strategy '{strategy.name}' "
                    f"on_candle: {exc}",
                    exc_info=True,
                )

    async def on_tick(self, tick: Tick) -> None:
        """
        Forward raw ticks to all registered strategies (optional).
        Most strategies ignore ticks and only process candles.
        """
        for strategy in self._strategies:
            try:
                signal = strategy.on_tick(tick)
                if signal is not None:
                    self._signal_counts[strategy.name] += 1
                    await self._dispatch_signal(signal)
            except Exception as exc:
                logger.error(
                    f"[StrategyEngine] Error in strategy '{strategy.name}' "
                    f"on_tick: {exc}",
                    exc_info=True,
                )

    async def _dispatch_signal(self, signal: TradeSignal) -> None:
        """Route signal to all registered handlers."""
        for handler in self._signal_handlers:
            try:
                await handler(signal)
            except Exception as exc:
                logger.error(
                    f"[StrategyEngine] Error in signal handler: {exc}",
                    exc_info=True,
                )

    def get_stats(self) -> dict:
        """Return per-strategy signal counts for monitoring."""
        return {
            "registered_strategies": [s.name for s in self._strategies],
            "signal_counts": dict(self._signal_counts),
            "total_signals": sum(self._signal_counts.values()),
        }
