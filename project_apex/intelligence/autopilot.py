"""
Project APEX — Autopilot Engine

The top-level autonomous orchestrator. Replaces static symbol lists with a
dynamic loop that:
  1. Rescores all symbols every N minutes via MarketSelector.
  2. Subscribes to the top-N symbols, unsubscribing from dropped ones.
  3. Routes incoming candles only for active (top-scoring) symbols.
  4. Logs every symbol switch with its scoring rationale.

This is the "brain" that decides WHERE to trade. The RiskEngine decides
WHETHER to trade. The strategies decide WHAT signal to emit.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Awaitable, Optional

from loguru import logger

from project_apex.intelligence.market_selector import MarketSelector
from project_apex.models.candle import Candle


class AutopilotEngine:
    """
    Autonomous market selection and feed management.

    Args:
        market_selector: Pre-configured MarketSelector instance.
        subscribe_fn: Async callable to subscribe to a symbol's tick feed.
        unsubscribe_fn: Async callable to unsubscribe from a symbol's tick feed.
        candle_handler: Async callable that receives candles for active symbols.
        top_n: Number of top symbols to actively trade at any time.
        rescore_interval_s: How often (seconds) to re-rank all symbols.
    """

    def __init__(
        self,
        market_selector: MarketSelector,
        subscribe_fn: Callable[[str], Awaitable[None]],
        unsubscribe_fn: Callable[[str], Awaitable[None]],
        candle_handler: Callable[[Candle], Awaitable[None]],
        top_n: int = 2,
        rescore_interval_s: float = 300.0,
    ) -> None:
        self._selector = market_selector
        self._subscribe = subscribe_fn
        self._unsubscribe = unsubscribe_fn
        self._candle_handler = candle_handler
        self._top_n = top_n
        self._rescore_interval = rescore_interval_s

        self._active_symbols: set[str] = set()
        self._rescore_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"[AutopilotEngine] Initialized | top_n={top_n} | "
            f"rescore_interval={rescore_interval_s}s"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the autopilot background scoring loop."""
        self._running = True
        # Run an initial scoring immediately before starting the loop
        await self._rescore_and_rebalance()
        self._rescore_task = asyncio.create_task(self._rescore_loop())
        logger.info("[AutopilotEngine] Started — autonomous mode engaged.")

    async def stop(self) -> None:
        """Stop the autopilot and unsubscribe from all active feeds."""
        self._running = False
        if self._rescore_task:
            self._rescore_task.cancel()
            try:
                await self._rescore_task
            except asyncio.CancelledError:
                pass

        for symbol in list(self._active_symbols):
            await self._unsubscribe(symbol)
        self._active_symbols.clear()
        logger.info("[AutopilotEngine] Stopped.")

    # ── Public API ────────────────────────────────────────────────────────────

    async def on_candle(self, candle: Candle) -> None:
        """
        Entry point for all incoming candles. Forwards only candles
        from currently active (top-scoring) symbols.
        """
        if candle.symbol in self._active_symbols:
            await self._candle_handler(candle)

    @property
    def active_symbols(self) -> list[str]:
        """The symbols currently being actively traded."""
        return list(self._active_symbols)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _rescore_loop(self) -> None:
        """Background task: re-rank symbols every rescore_interval seconds."""
        while self._running:
            await asyncio.sleep(self._rescore_interval)
            try:
                await self._rescore_and_rebalance()
            except Exception as exc:
                logger.error(f"[AutopilotEngine] Rescore error: {exc}")

    async def _rescore_and_rebalance(self) -> None:
        """Score all symbols and subscribe/unsubscribe as needed."""
        scores = await self._selector.update_all()
        new_top = set(self._selector.get_top_symbols(self._top_n))

        # Unsubscribe symbols that dropped out of the top-N
        to_remove = self._active_symbols - new_top
        for symbol in to_remove:
            score_obj = next((s for s in scores if s.symbol == symbol), None)
            logger.info(
                f"[AutopilotEngine] ⬇ DROPPING {symbol} "
                f"(score={score_obj.total_score:.1f if score_obj else '?'})"
            )
            try:
                await self._unsubscribe(symbol)
            except Exception as exc:
                logger.warning(f"[AutopilotEngine] Unsubscribe error for {symbol}: {exc}")
            self._active_symbols.discard(symbol)

        # Subscribe to new symbols that entered the top-N
        to_add = new_top - self._active_symbols
        for symbol in to_add:
            score_obj = next((s for s in scores if s.symbol == symbol), None)
            logger.info(
                f"[AutopilotEngine] ⬆ SELECTING {symbol} | "
                f"score={score_obj.total_score:.1f if score_obj else '?'} | "
                f"regime={score_obj.regime if score_obj else '?'} | "
                f"ADX={score_obj.adx if score_obj else '?'}"
            )
            try:
                await self._subscribe(symbol)
                self._active_symbols.add(symbol)
            except Exception as exc:
                logger.warning(f"[AutopilotEngine] Subscribe error for {symbol}: {exc}")

        if not to_add and not to_remove:
            logger.debug(
                f"[AutopilotEngine] No symbol change. Active: {sorted(self._active_symbols)}"
            )
