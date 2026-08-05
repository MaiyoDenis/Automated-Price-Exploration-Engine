"""
Project APEX
Application Core — Elite Autonomous Mode

Wires all services in dependency order and orchestrates the full autonomous
trading loop:

  Database → Repository → Deriv Client → Candle Builder → Tick Collector
    → Portfolio → Broker → CircuitBreaker → Risk Engine
    → Strategy Engine (MetaRegimeStrategy + Ensemble + ML)
    → MarketSelector → AutopilotEngine
    → ModelTrainer → AlertManager → HealthMonitor
    → Dashboard → run loop
"""
from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd
from loguru import logger

from project_apex.config.config import Config
from project_apex.config.environment import Environment
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.database.sqlite_market_data import SQLiteMarketDataRepository
from project_apex.api.deriv_client import DerivClient
from project_apex.api.messages import MessageBuilder
from project_apex.data.candle_builder import CandleBuilder
from project_apex.data.tick_processor import TickCollector
from project_apex.utils.logger import setup_logger

# Strategy layer
from project_apex.core.strategy_engine import StrategyEngine
from project_apex.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from project_apex.strategies.macd_momentum import MACDMomentumStrategy
from project_apex.strategies.bollinger_breakout import BollingerBreakoutStrategy
from project_apex.strategies.multi_strategy import MultiStrategyEnsemble
from project_apex.strategies.ml_strategy import MLStrategy
from project_apex.strategies.meta_strategy import MetaRegimeStrategy

# Risk layer
from project_apex.risk.engine import RiskEngine
from project_apex.risk.circuit_breaker import CircuitBreaker

# Execution layer
from project_apex.execution.paper_broker import PaperBroker
from project_apex.execution.live_broker import LiveBroker
from project_apex.execution.portfolio import Portfolio
from project_apex.execution.models import Trade
from project_apex.risk.models import TradeOrder

# Intelligence layer
from project_apex.intelligence.market_selector import MarketSelector, DERIV_UNIVERSE
from project_apex.intelligence.autopilot import AutopilotEngine

# AI / ML layer
from project_apex.ai.models import XGBoostPredictor
from project_apex.ai.trainer import ModelTrainer

# Monitoring layer
from project_apex.monitoring.alerting import AlertManager
from project_apex.monitoring.health_monitor import HealthMonitor

# Dashboard
from project_apex.dashboard.server import DashboardServer

# Models
from project_apex.models.candle import Candle
from project_apex.models.tick import Tick


class Application:
    """Coordinates all application services in autonomous elite mode."""

    def __init__(self, paper_trading: bool = True) -> None:
        setup_logger()
        logger.info("Starting Project APEX — Elite Autonomous Mode...")

        self.config = Config()
        self.environment = Environment()
        self.paper_trading = paper_trading
        logger.info(f"Mode: {'Paper Trading' if paper_trading else 'LIVE Trading'}")

        # Service references
        self.database: Optional[SQLiteManager] = None
        self.repository: Optional[SQLiteMarketDataRepository] = None
        self.deriv_client: Optional[DerivClient] = None
        self.candle_builder: Optional[CandleBuilder] = None
        self.tick_collector: Optional[TickCollector] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.ensemble: Optional[MultiStrategyEnsemble] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.broker: Optional[PaperBroker] = None
        self.portfolio: Optional[Portfolio] = None
        self.market_selector: Optional[MarketSelector] = None
        self.autopilot: Optional[AutopilotEngine] = None
        self.model_trainer: Optional[ModelTrainer] = None
        self.alert_manager: Optional[AlertManager] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.dashboard: Optional[DashboardServer] = None

    async def initialize(self) -> None:
        """Start all required services in dependency order."""
        logger.info("Initializing services...")

        # ── 1. Database & Repository ──────────────────────────────────────────
        db_path = self.config.get_str("database", "path")
        self.database = SQLiteManager(db_path)
        self.database.connect()
        self.repository = SQLiteMarketDataRepository(self.database)
        self.repository.initialize()

        # ── 2. Alert Manager (early — used by all subsequent layers) ──────────
        self.alert_manager = AlertManager()
        self.alert_manager.system("Project APEX starting up — autonomous mode.")

        # ── 3. Portfolio & Broker ─────────────────────────────────────────────
        initial_capital = float(self.config.get("trading", "initial_capital") or 10_000.0)
        self.portfolio = Portfolio(initial_capital=initial_capital)

        if self.paper_trading:
            slippage = float(self.config.get("trading", "slippage_pct") or 0.0001)
            spread = float(self.config.get("trading", "spread_pct") or 0.0002)
            self.broker = PaperBroker(slippage_pct=slippage, spread_pct=spread)
        else:
            self.broker = LiveBroker()

        self.broker.add_trade_handler(self._on_trade_closed)

        # ── 4. Circuit Breaker ────────────────────────────────────────────────
        self.circuit_breaker = CircuitBreaker(
            halt_on_consecutive_sl=int(self.config.get("risk", "circuit_breaker_sl_streak") or 3),
            loss_velocity_pct=float(self.config.get("risk", "circuit_breaker_loss_vel_pct") or 0.02),
            velocity_window_s=float(self.config.get("risk", "circuit_breaker_velocity_window_s") or 1800.0),
            max_trades_per_hour=int(self.config.get("risk", "circuit_breaker_max_trades_hr") or 10),
            halt_duration_s=float(self.config.get("risk", "circuit_breaker_halt_duration_s") or 3600.0),
            db=self.database,  # enables persistent halt state across restarts
        )

        # ── 5. Risk Engine ────────────────────────────────────────────────────
        self.risk_engine = RiskEngine(
            max_daily_loss_pct=float(self.config.get("risk", "max_daily_loss_pct") or 0.05),
            max_drawdown_pct=float(self.config.get("risk", "max_drawdown_pct") or 0.15),
            max_open_positions=int(self.config.get("risk", "max_open_positions") or 3),
            symbol_cooldown_s=float(self.config.get("risk", "symbol_cooldown_s") or 60.0),
            risk_per_trade_pct=float(self.config.get("risk", "risk_per_trade_pct") or 0.01),
            kelly_fraction=float(self.config.get("risk", "kelly_fraction") or 0.25),
            min_confidence=float(self.config.get("risk", "min_confidence") or 0.3),
        )
        # Wire circuit breaker into risk engine
        self.risk_engine.set_circuit_breaker(self.circuit_breaker)

        # Inject live portfolio state getters
        getters = self.portfolio.get_risk_getters()
        self.risk_engine.set_portfolio_getters(**getters)
        self.risk_engine.add_order_handler(self._on_order_approved)

        # ── 6. Strategy Engine + MetaRegime Ensemble ──────────────────────────
        timeframes = self.config.get_list("market", "timeframes")
        primary_tf = timeframes[0]
        secondary_tf = timeframes[1] if len(timeframes) > 1 else primary_tf

        rsi_strategy = RSIMeanReversionStrategy(
            name="RSI_MeanReversion",
            config={"rsi_period": 14, "oversold": 30, "overbought": 70,
                    "min_adx": 20, "timeframe": primary_tf},
        )
        macd_strategy = MACDMomentumStrategy(
            name="MACD_Momentum",
            config={"timeframe": secondary_tf},
        )
        bb_strategy = BollingerBreakoutStrategy(
            name="Bollinger_Breakout",
            config={"timeframe": timeframes[-1]},
        )

        self.ensemble = MultiStrategyEnsemble(
            strategies=[rsi_strategy, macd_strategy, bb_strategy],
            min_vote_fraction=0.55,
            min_strategies_agree=2,
            name="ApexEnsemble",
            db=self.database,  # enables persistent strategy performance stats
        )

        # ML strategy as a parallel voter in the ensemble
        ml_strategy = MLStrategy(
            name="ML_XGBoost",
            config={"timeframe": primary_tf, "model_path": "datasets/xgb_model.joblib"},
        )

        # MetaRegimeStrategy routes to the right strategies based on regime
        meta_strategy = MetaRegimeStrategy(
            trend_strategies=[macd_strategy, ml_strategy],
            ranging_strategies=[rsi_strategy, bb_strategy],
            name="MetaRegimeRouter",
            config={"timeframe": primary_tf},
        )

        self.strategy_engine = StrategyEngine()
        self.strategy_engine.register(meta_strategy)
        self.strategy_engine.register(self.ensemble)
        self.strategy_engine.add_signal_handler(self.risk_engine.evaluate)

        # ── 7. ML Auto-Trainer ────────────────────────────────────────────────
        predictor = ml_strategy.predictor if hasattr(ml_strategy, "predictor") else XGBoostPredictor()
        self.model_trainer = ModelTrainer(
            predictor=predictor,
            db=self.database,
            symbol=self.config.get_list("market", "symbols")[0] if self.config.get_list("market", "symbols") else "R_50",
            timeframe=primary_tf,
            retrain_interval_h=float(self.config.get("ml", "retrain_interval_h") or 24.0),
            drift_check_interval_m=float(self.config.get("ml", "drift_check_interval_m") or 60.0),
        )

        # ── 8. DerivClient ────────────────────────────────────────────────────
        self.deriv_client = DerivClient(self.config)

        # ── 9. CandleBuilder ──────────────────────────────────────────────────
        self.candle_builder = CandleBuilder(
            timeframes=timeframes,
            valid_timeframes=timeframes,
        )
        self.candle_builder.register_candle_callback(self._on_candle_completed)

        # ── 10. Market Selector + Autopilot ───────────────────────────────────
        self.market_selector = MarketSelector(
            candle_fetcher=self._fetch_candles_for_selector,
            universe=self.config.get_list("market", "universe") or DERIV_UNIVERSE,
            timeframe=int(self.config.get("intelligence", "selector_timeframe") or 300),
            cooldown_s=float(self.config.get("intelligence", "cooldown_s") or 1800.0),
        )

        initial_symbols = self.config.get_list("market", "symbols")

        self.autopilot = AutopilotEngine(
            market_selector=self.market_selector,
            subscribe_fn=self._subscribe_symbol,
            unsubscribe_fn=self._unsubscribe_symbol,
            candle_handler=self._on_candle_from_autopilot,
            top_n=int(self.config.get("intelligence", "top_n_symbols") or 2),
            rescore_interval_s=float(self.config.get("intelligence", "rescore_interval_s") or 300.0),
        )

        # ── 11. TickCollector ─────────────────────────────────────────────────
        stats_interval = self.config.get_int("market", "stats_interval")
        self.tick_collector = TickCollector(
            provider=self.deriv_client,
            repository=self.repository,
            candle_builder=self.candle_builder,
            symbols=initial_symbols,
            stats_interval=stats_interval,
            valid_timeframes=timeframes,
        )

        # ── 12. Health Monitor ────────────────────────────────────────────────
        self.health_monitor = HealthMonitor(app=self)
        await self.health_monitor.start()

        # ── 13. Dashboard Server ──────────────────────────────────────────────
        self.dashboard = DashboardServer(self)
        await self.dashboard.start()

        # ── 14. Connect & Start ───────────────────────────────────────────────
        await self.deriv_client.connect()
        await self.tick_collector.start()
        await self.autopilot.start()
        await self.model_trainer.start()

        logger.success("Application initialized successfully — Autonomous mode ACTIVE.")
        logger.info(f"Initial symbols: {initial_symbols}")
        logger.info(f"Timeframes: {timeframes}s")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def _on_candle_completed(self, candle: Candle) -> None:
        """Candle from the builder → forward to autopilot (which filters by active symbols)."""
        # Update health monitor
        if self.health_monitor:
            self.health_monitor.record_tick(candle.symbol)

        # Update portfolio price for unrealized PnL
        if self.portfolio:
            self.portfolio.update_price(candle.symbol, candle.close)

        # Update circuit breaker with latest equity
        if self.circuit_breaker and self.portfolio:
            self.circuit_breaker.update_equity(self.portfolio.equity)

        # Route to autopilot (only active symbols proceed)
        if self.autopilot:
            await self.autopilot.on_candle(candle)

        # Check stop/target hits on every candle close
        if self.broker:
            await self.broker.check_stops_and_targets(
                symbol=candle.symbol,
                current_price=candle.close,
                current_time_ms=candle.timestamp,
            )

    async def _on_candle_from_autopilot(self, candle: Candle) -> None:
        """Candle approved by autopilot → send to strategy engine."""
        await self.strategy_engine.on_candle(candle)

    async def _on_order_approved(self, order: TradeOrder) -> None:
        """RiskEngine approved → open position."""
        if self.broker is None or self.portfolio is None:
            return

        position = await self.broker.open_position(order)
        self.portfolio.on_position_opened(position)

        # Circuit breaker: record trade opened
        if self.circuit_breaker:
            self.circuit_breaker.record_trade_opened()

        # Alert
        if self.alert_manager:
            self.alert_manager.trade_opened(
                symbol=order.symbol,
                direction=order.direction.name,
                size=order.size,
                price=order.entry_price,
                strategy=order.strategy_name,
            )

        summary = self.portfolio.summary()
        logger.info(
            f"[Portfolio] Equity=${summary['equity']:,.2f} "
            f"Positions={summary['open_positions']} "
            f"Daily={summary['daily_pnl_pct']:+.2f}% "
            f"Drawdown={summary['drawdown_pct']:.2f}%"
        )

    async def _on_trade_closed(self, trade: Trade) -> None:
        """Broker closed a position → update everything."""
        if self.portfolio:
            self.portfolio.on_trade_closed(trade)

        is_stop = trade.close_reason == "stop_loss"

        # Circuit breaker: record trade outcome
        if self.circuit_breaker and self.portfolio:
            self.circuit_breaker.record_trade_closed(
                is_stop_loss=is_stop,
                equity=self.portfolio.equity,
            )

        # Market selector: apply cooldown on stop-loss
        if is_stop and self.market_selector:
            self.market_selector.record_stop_loss(trade.symbol)

        # Update adaptive strategy weights in ensemble
        if self.ensemble:
            self.ensemble.record_trade_result(
                strategy_name=trade.strategy_name,
                pnl=trade.realized_pnl,
                pnl_pct=trade.realized_pnl_pct,
            )

        # Alert
        if self.alert_manager:
            self.alert_manager.trade_closed(
                symbol=trade.symbol,
                direction=trade.direction.name,
                pnl=trade.realized_pnl,
                pnl_pct=trade.realized_pnl_pct,
                reason=trade.close_reason,
                strategy=trade.strategy_name,
            )

    async def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to tick feed for a symbol."""
        if self.tick_collector:
            await self.tick_collector.subscribe(symbol)
        logger.info(f"[Application] Subscribed to {symbol}")

    async def _unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from tick feed for a symbol."""
        if self.tick_collector:
            await self.tick_collector.unsubscribe(symbol)
        logger.info(f"[Application] Unsubscribed from {symbol}")

    async def _fetch_candles_for_selector(self, symbol: str, timeframe: int) -> pd.DataFrame:
        """Fetch recent candle data for the MarketSelector scoring."""
        if self.repository is None:
            return pd.DataFrame()
        try:
            import time
            start_ts = int(time.time()) - 86400 * 7  # Last 7 days
            candles = self.repository.get_candles(symbol, timeframe, start_ts, int(time.time()))
            if not candles:
                return pd.DataFrame()
            return pd.DataFrame([
                {"timestamp": c.timestamp, "open": c.open, "high": c.high,
                 "low": c.low, "close": c.close}
                for c in candles
            ])
        except Exception:
            return pd.DataFrame()

    # ── Shutdown ───────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Cleanly stop all services."""
        logger.info("Shutting down services...")

        if self.autopilot:
            await self.autopilot.stop()

        if self.model_trainer:
            await self.model_trainer.stop()

        if self.health_monitor:
            await self.health_monitor.stop()

        if self.dashboard:
            await self.dashboard.stop()

        if self.tick_collector:
            await self.tick_collector.stop()

        if self.deriv_client:
            await self.deriv_client.disconnect()

        if self.portfolio:
            summary = self.portfolio.summary()
            logger.info(
                f"Final Portfolio | "
                f"Equity=${summary['equity']:,.2f} "
                f"Return={summary['total_return_pct']:+.2f}% "
                f"Trades={summary['closed_trades']} "
                f"WinRate={summary['win_rate_pct']:.1f}% "
                f"MaxDD={summary['drawdown_pct']:.2f}%"
            )
            if self.alert_manager:
                self.alert_manager.daily_summary(
                    equity=summary["equity"],
                    return_pct=summary["total_return_pct"] / 100,
                    trades=summary["closed_trades"],
                    win_rate=summary["win_rate_pct"] / 100,
                    drawdown_pct=summary["drawdown_pct"] / 100,
                )

        if self.database:
            self.database.close()

        logger.success("Application stopped successfully.")

    async def run(self) -> None:
        """Run the application until interrupted."""
        try:
            await self.initialize()
            while True:
                await asyncio.sleep(3600)
                if self.strategy_engine:
                    stats = self.strategy_engine.get_stats()
                    logger.info(f"[StrategyEngine] Stats: {stats}")
                if self.portfolio:
                    logger.info(f"[Portfolio] {self.portfolio.summary()}")
                if self.ensemble:
                    perf = self.ensemble.get_performance_summary()
                    logger.info(f"[StrategyPerformance] {perf}")
        finally:
            await self.shutdown()