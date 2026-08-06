"""
Project APEX — Dashboard REST API Server (Elite Upgrade)

REST endpoints:
  GET /                         — Dashboard UI
  GET /api/status               — System status
  GET /api/portfolio            — Live portfolio summary
  GET /api/positions            — Open positions
  GET /api/market-scores        — Symbol opportunity rankings
  GET /api/strategy-performance — Per-strategy live stats
  GET /api/model-info           — ML model status and trainer info
  GET /api/alerts               — Recent system alerts (ring buffer)
  GET /api/health               — System health checks
  GET /api/trades               — Closed trade history (paginated)
  GET /api/strategies           — Strategy engine stats
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TYPE_CHECKING

from aiohttp import web
from loguru import logger

if TYPE_CHECKING:
    from project_apex.core.application import Application


class DashboardServer:
    """Elite REST API for the APEX dashboard."""

    def __init__(self, app_core: "Application", host: str = "127.0.0.1", port: int = 8080) -> None:
        self.app_core = app_core
        self.host = host
        self.port = port
        self.app = web.Application()
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self) -> None:
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/portfolio", self.handle_portfolio)
        self.app.router.add_get("/api/positions", self.handle_positions)
        self.app.router.add_get("/api/strategies", self.handle_strategies)
        self.app.router.add_get("/api/market-scores", self.handle_market_scores)
        self.app.router.add_get("/api/strategy-performance", self.handle_strategy_performance)
        self.app.router.add_get("/api/model-info", self.handle_model_info)
        self.app.router.add_get("/api/alerts", self.handle_alerts)
        self.app.router.add_get("/api/health", self.handle_health)
        self.app.router.add_get("/api/trades", self.handle_trades)
        
        self.app.router.add_post("/api/settings/mode", self.handle_set_mode)
        self.app.router.add_post("/api/settings/account", self.handle_set_account)
        self.app.router.add_get("/api/account-balance", self.handle_account_balance)
        self.app.router.add_get("/api/settings/trading-params", self.handle_get_trading_params)
        self.app.router.add_post("/api/settings/trading-params", self.handle_set_trading_params)

        self.app.router.add_get("/", self.handle_index)
        if os.path.isdir(self.static_dir):
            self.app.router.add_static("/static/", path=self.static_dir, name="static")

    # ── Handlers ──────────────────────────────────────────────────────────────

    async def handle_index(self, request: web.Request) -> web.Response:
        index_path = os.path.join(self.static_dir, "index.html")
        if os.path.exists(index_path):
            return web.FileResponse(index_path)
        return web.Response(text="<h1>APEX Dashboard — UI not found</h1>", content_type="text/html")

    async def handle_status(self, request: web.Request) -> web.Response:
        is_connected = False
        if self.app_core.deriv_client:
            is_connected = self.app_core.deriv_client.is_connected

        active_symbols = []
        if self.app_core.autopilot:
            active_symbols = self.app_core.autopilot.active_symbols

        cb_halted = False
        cb_reason = ""
        if self.app_core.circuit_breaker:
            cb_halted = self.app_core.circuit_breaker.is_halted
            cb_reason = self.app_core.circuit_breaker.halt_reason

        risk_halted = False
        if self.app_core.risk_engine:
            risk_halted = self.app_core.risk_engine._trading_halt

        return await self._json({
            "status": "online",
            "broker_connected": is_connected,
            "account_type": self.app_core.config.get("api", "account_type") or "demo",
            "trading_mode": getattr(self.app_core, "trading_mode", "standard"),
            "active_symbols": active_symbols,
            "circuit_breaker_halted": cb_halted,
            "circuit_breaker_reason": cb_reason,
            "risk_engine_halted": risk_halted,
        })

    async def handle_portfolio(self, request: web.Request) -> web.Response:
        portfolio = self.app_core.portfolio
        if portfolio is None:
            return await self._json({"error": "Portfolio not initialized"})
        return await self._json(portfolio.summary())

    async def handle_positions(self, request: web.Request) -> web.Response:
        portfolio = self.app_core.portfolio
        if portfolio is None:
            return await self._json({"error": "Portfolio not initialized"})

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
                "unrealized_pnl": round(pos.unrealized_pnl(curr_price), 4),
                "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(curr_price) * 100, 2),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "opened_at": pos.opened_at,
                "strategy": pos.strategy_name,
            })
        return await self._json({"positions": positions, "count": len(positions)})

    async def handle_strategies(self, request: web.Request) -> web.Response:
        engine = self.app_core.strategy_engine
        if engine is None:
            return await self._json({"error": "Strategy engine not initialized"})
        return await self._json(engine.get_stats())

    async def handle_market_scores(self, request: web.Request) -> web.Response:
        selector = self.app_core.market_selector
        if selector is None:
            return await self._json({"error": "Market selector not initialized"})

        scores = selector.get_all_scores()
        active = self.app_core.autopilot.active_symbols if self.app_core.autopilot else []
        for s in scores:
            s["is_active"] = s["symbol"] in active
        return await self._json({"scores": scores, "active_symbols": active})

    async def handle_strategy_performance(self, request: web.Request) -> web.Response:
        ensemble = self.app_core.ensemble
        if ensemble is None:
            return await self._json({"error": "Ensemble not initialized"})
        return await self._json({"strategies": ensemble.get_performance_summary()})

    async def handle_model_info(self, request: web.Request) -> web.Response:
        trainer = self.app_core.model_trainer
        if trainer is None:
            return await self._json({"error": "Model trainer not initialized"})

        predictor = trainer._predictor
        return await self._json({
            "is_trained": predictor.is_trained,
            "model_path": predictor.model_path,
            "symbol": trainer._symbol,
            "timeframe": trainer._timeframe,
            "lookback_days": trainer._lookback_days,
            "retrain_interval_h": trainer._retrain_interval_s / 3600,
            "drift_check_interval_m": trainer._drift_check_interval_s / 60,
            "has_training_baseline": trainer._training_return_distribution is not None,
        })

    async def handle_alerts(self, request: web.Request) -> web.Response:
        alert_manager = self.app_core.alert_manager
        if alert_manager is None:
            return await self._json({"alerts": [], "count": 0})

        n = int(request.rel_url.query.get("n", 50))
        category = request.rel_url.query.get("category")

        if category:
            alerts = alert_manager.get_by_category(category, n=n)
        else:
            alerts = alert_manager.get_recent(n=n)

        return await self._json({"alerts": alerts, "count": len(alerts)})

    async def handle_health(self, request: web.Request) -> web.Response:
        health_monitor = self.app_core.health_monitor
        if health_monitor is None:
            return await self._json({"overall": "UNKNOWN", "checks": []})
        return await self._json(health_monitor.get_health_report())

    async def handle_trades(self, request: web.Request) -> web.Response:
        portfolio = self.app_core.portfolio
        if portfolio is None:
            return await self._json({"trades": [], "count": 0})

        n = int(request.rel_url.query.get("n", 50))
        trades = portfolio._closed_trades[-n:]
        trades_data = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "direction": t.direction.name,
                "size": t.size,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.realized_pnl, 4),
                "pnl_pct": round(t.realized_pnl_pct * 100, 2),
                "close_reason": t.close_reason,
                "strategy": t.strategy_name,
                "opened_at": t.opened_at,
                "closed_at": t.closed_at,
            }
            for t in reversed(trades)
        ]
        return await self._json({
            "trades": trades_data,
            "count": len(portfolio._closed_trades),
        })

    async def handle_set_mode(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            mode = data.get("mode")
            if mode not in ("standard", "scalping", "differs"):
                return await self._json({"error": "Invalid mode. Must be 'standard', 'scalping', or 'differs'."})
            
            if hasattr(self.app_core, "set_trading_mode"):
                self.app_core.set_trading_mode(mode)
                return await self._json({"status": "success", "mode": mode})
            else:
                return await self._json({"error": "Core application does not support mode switching."})
        except json.JSONDecodeError:
            return await self._json({"error": "Invalid JSON"})

    async def handle_set_account(self, request: web.Request) -> web.Response:
        """Switch the active trading account between demo and real."""
        try:
            data = await request.json()
            account_type = data.get("type")
            if account_type not in ("demo", "real"):
                return await self._json({"error": "Invalid type. Must be 'demo' or 'real'."})

            if not hasattr(self.app_core, "switch_account"):
                return await self._json({"error": "Account switching not supported."})

            result = await self.app_core.switch_account(account_type)
            return await self._json(result)
        except json.JSONDecodeError:
            return await self._json({"error": "Invalid JSON"})
        except Exception as e:
            logger.error(f"[DashboardServer] Account switch error: {e}")
            return await self._json({"error": str(e)})

    async def handle_account_balance(self, request: web.Request) -> web.Response:
        account_type = request.rel_url.query.get("type", "demo")
        if self.app_core.deriv_client is None:
            return await self._json({"error": "Deriv client not initialized"})
        result = await self.app_core.deriv_client.get_account_balance(account_type)
        return await self._json(result)

    async def handle_get_trading_params(self, request: web.Request) -> web.Response:
        """Return current trading parameters for all modes."""
        params = {"differs": {}, "scalping": {}, "standard": {}}

        dre = getattr(self.app_core, "differs_risk_engine", None)
        if dre:
            params["differs"] = {
                "base_stake": dre.base_stake,
                "max_stake": dre.max_stake,
                "martingale_enabled": dre.martingale_enabled,
                "martingale_multiplier": dre.martingale_multiplier,
                "martingale_max_steps": dre.martingale_max_steps,
                "max_daily_loss_pct": dre.max_daily_loss_pct,
                "loss_cooldown_ticks": dre.loss_cooldown_ticks,
            }

        sre = getattr(self.app_core, "scalping_risk_engine", None)
        if sre:
            params["scalping"] = {
                "risk_per_trade_pct": sre.risk_per_trade_pct,
                "atr_stop_multiplier": sre.atr_stop_multiplier,
                "atr_tp_multiplier": sre.atr_tp_multiplier,
                "max_open_positions": sre.max_open_positions,
                "max_daily_loss_pct": sre.max_daily_loss_pct,
            }

        stdre = getattr(self.app_core, "standard_risk_engine", None)
        if stdre:
            params["standard"] = {
                "risk_per_trade_pct": stdre.risk_per_trade_pct,
                "atr_stop_multiplier": stdre.atr_stop_multiplier,
                "atr_tp_multiplier": stdre.atr_tp_multiplier,
                "max_open_positions": stdre.max_open_positions,
                "max_daily_loss_pct": stdre.max_daily_loss_pct,
                "min_confidence": stdre.min_confidence,
            }

        return await self._json(params)

    async def handle_set_trading_params(self, request: web.Request) -> web.Response:
        """Update trading parameters for a specific mode in real-time."""
        try:
            data = await request.json()
            mode = data.get("mode")
            params = data.get("params", {})

            if mode not in ("differs", "scalping", "standard"):
                return await self._json({"error": "Invalid mode"})

            if mode == "differs":
                engine = getattr(self.app_core, "differs_risk_engine", None)
                if not engine:
                    return await self._json({"error": "Differs risk engine not initialized"})
                _DIFFERS_FIELDS = {
                    "base_stake": float, "max_stake": float,
                    "martingale_enabled": bool, "martingale_multiplier": float,
                    "martingale_max_steps": int,
                    "max_daily_loss_pct": float, "loss_cooldown_ticks": int,
                }
                for key, cast in _DIFFERS_FIELDS.items():
                    if key in params:
                        setattr(engine, key, cast(params[key]))

            elif mode in ("scalping", "standard"):
                attr = "scalping_risk_engine" if mode == "scalping" else "standard_risk_engine"
                engine = getattr(self.app_core, attr, None)
                if not engine:
                    return await self._json({"error": f"{mode} risk engine not initialized"})
                _STD_FIELDS = {
                    "risk_per_trade_pct": float, "atr_stop_multiplier": float,
                    "atr_tp_multiplier": float, "max_open_positions": int,
                    "max_daily_loss_pct": float, "min_confidence": float,
                }
                for key, cast in _STD_FIELDS.items():
                    if key in params:
                        setattr(engine, key, cast(params[key]))

            logger.info(f"[DashboardServer] Trading params updated for {mode}: {params}")
            return await self._json({"status": "success", "mode": mode})
        except json.JSONDecodeError:
            return await self._json({"error": "Invalid JSON"})
        except Exception as e:
            logger.error(f"[DashboardServer] Settings update error: {e}")
            return await self._json({"error": str(e)})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"[DashboardServer] Listening on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            logger.info("[DashboardServer] Stopped")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _json(self, data: dict[str, Any]) -> web.Response:
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Cache-Control": "no-store",
        }
        return web.json_response(data, headers=headers)
