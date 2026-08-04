"""
Project APEX — Dashboard REST API Server

Serves endpoints for the frontend to query portfolio status, trades, and signals.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from aiohttp import web
from loguru import logger

from project_apex.core.application import Application
from project_apex.execution.portfolio import Portfolio
from project_apex.core.strategy_engine import StrategyEngine


class DashboardServer:
    """
    REST API for the APEX dashboard.
    """

    def __init__(self, app_core: Application, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.app_core = app_core
        self.host = host
        self.port = port
        self.app = web.Application()
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self) -> None:
        # API Routes
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/portfolio", self.handle_portfolio)
        self.app.router.add_get("/api/positions", self.handle_positions)
        self.app.router.add_get("/api/strategies", self.handle_strategies)
        
        # Static UI Routes
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_static("/static/", path=self.static_dir, name="static")

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serves the main dashboard HTML."""
        index_path = os.path.join(self.static_dir, "index.html")
        return web.FileResponse(index_path)

    async def _cors_response(self, data: dict[str, Any]) -> web.Response:
        """Returns a JSON response with permissive CORS headers for local dev."""
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Cache-Control": "no-store",
        }
        return web.json_response(data, headers=headers)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Returns overall system status."""
        is_connected = False
        if self.app_core.deriv_client:
            is_connected = self.app_core.deriv_client.is_connected

        return await self._cors_response({
            "status": "online",
            "broker_connected": is_connected,
            "mode": "paper" if self.app_core.paper_trading else "live"
        })

    async def handle_portfolio(self, request: web.Request) -> web.Response:
        """Returns portfolio summary and metrics."""
        portfolio: Portfolio | None = self.app_core.portfolio
        if portfolio is None:
            return await self._cors_response({"error": "Portfolio not initialized"})

        return await self._cors_response(portfolio.summary())

    async def handle_positions(self, request: web.Request) -> web.Response:
        """Returns active open positions."""
        portfolio: Portfolio | None = self.app_core.portfolio
        if portfolio is None:
            return await self._cors_response({"error": "Portfolio not initialized"})

        # Use latest prices for accurate unrealized PnL
        prices = portfolio._latest_price
        
        positions = []
        for pos_id, pos in portfolio._open_positions.items():
            curr_price = prices.get(pos.symbol, pos.entry_price)
            positions.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "direction": pos.direction.name,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": curr_price,
                "unrealized_pnl": pos.unrealized_pnl(curr_price),
                "unrealized_pnl_pct": pos.unrealized_pnl_pct(curr_price),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "opened_at": pos.opened_at,
                "strategy": pos.strategy_name
            })
            
        return await self._cors_response({"positions": positions})

    async def handle_strategies(self, request: web.Request) -> web.Response:
        """Returns strategy engine statistics."""
        engine: StrategyEngine | None = self.app_core.strategy_engine
        if engine is None:
            return await self._cors_response({"error": "Strategy engine not initialized"})

        return await self._cors_response(engine.get_stats())

    async def start(self) -> None:
        """Start the aiohttp server in the background."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"[DashboardServer] Listening on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the aiohttp server."""
        if self._runner is not None:
            await self._runner.cleanup()
            logger.info("[DashboardServer] Stopped")
