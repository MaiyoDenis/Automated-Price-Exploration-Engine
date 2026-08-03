"""
Project APEX
Application Core

Coordinates the startup and shutdown of all services, wired in dependency order:
  Database → Repository → Deriv Client → Candle Builder → Tick Collector
      → Portfolio → Broker → Risk Engine → Strategy Engine → Application run loop
"""
from __future__ import annotations

import asyncio
from typing import Optional

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

# Phase 3 — Strategy Engine
from project_apex.core.strategy_engine import StrategyEngine

# Phase 3 — Live Strategies
from project_apex.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from project_apex.strategies.macd_momentum import MACDMomentumStrategy
from project_apex.strategies.bollinger_breakout import BollingerBreakoutStrategy
from project_apex.strategies.multi_strategy import MultiStrategyEnsemble

# Phase 4 — Risk Engine
from project_apex.risk.engine import RiskEngine

# Phase 5 — Execution + Portfolio
from project_apex.execution.paper_broker import PaperBroker
from project_apex.execution.live_broker import LiveBroker
from project_apex.execution.portfolio import Portfolio
from project_apex.execution.models import Trade
from project_apex.risk.models import TradeOrder

# Phase 5B - Dashboard
from project_apex.dashboard.server import DashboardServer

# Models
from project_apex.models.candle import Candle
from project_apex.models.tick import Tick


class Application:
    """Coordinates all application services."""

    def __init__(self, paper_trading: bool = True) -> None:
        # Configure logging first
        setup_logger()

        logger.info("Starting Project APEX...")

        # Load configuration
        self.config = Config()
        self.environment = Environment()
        self.paper_trading = paper_trading
        logger.info(f"Mode: {'Paper Trading' if paper_trading else 'LIVE Trading'}")

        # Service references (set during initialize)
        self.database: Optional[SQLiteManager] = None
        self.repository: Optional[SQLiteMarketDataRepository] = None
        self.deriv_client: Optional[DerivClient] = None
        self.candle_builder: Optional[CandleBuilder] = None
        self.tick_collector: Optional[TickCollector] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.broker: Optional[PaperBroker] = None
        self.portfolio: Optional[Portfolio] = None
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

        # ── 2. Portfolio & Broker ─────────────────────────────────────────────
        initial_capital = float(self.config.get("trading", "initial_capital") or 10_000.0)
        self.portfolio = Portfolio(initial_capital=initial_capital)

        if self.paper_trading:
            slippage = float(self.config.get("trading", "slippage_pct") or 0.0001)
            spread = float(self.config.get("trading", "spread_pct") or 0.0002)
            self.broker = PaperBroker(slippage_pct=slippage, spread_pct=spread)
        else:
            self.broker = LiveBroker()

        self.broker.add_trade_handler(self._on_trade_closed)

        # ── 3. Risk Engine ────────────────────────────────────────────────────
        self.risk_engine = RiskEngine(
            max_daily_loss_pct=float(self.config.get("risk", "max_daily_loss_pct") or 0.05),
            max_drawdown_pct=float(self.config.get("risk", "max_drawdown_pct") or 0.15),
            max_open_positions=int(self.config.get("risk", "max_open_positions") or 3),
            symbol_cooldown_s=float(self.config.get("risk", "symbol_cooldown_s") or 60.0),
            risk_per_trade_pct=float(self.config.get("risk", "risk_per_trade_pct") or 0.01),
            min_confidence=float(self.config.get("risk", "min_confidence") or 0.3),
        )
        # Inject live portfolio state getters
        getters = self.portfolio.get_risk_getters()
        self.risk_engine.set_portfolio_getters(**getters)
        self.risk_engine.add_order_handler(self._on_order_approved)

        # ── 4. Strategy Engine + Strategies ───────────────────────────────────
        timeframes = self.config.get_list("market", "timeframes")

        rsi_strategy = RSIMeanReversionStrategy(
            name="RSI_MeanReversion",
            config={"rsi_period": 14, "oversold": 30, "overbought": 70,
                    "min_adx": 20, "timeframe": timeframes[0]},
        )
        macd_strategy = MACDMomentumStrategy(
            name="MACD_Momentum",
            config={"timeframe": timeframes[1] if len(timeframes) > 1 else timeframes[0]},
        )
        bb_strategy = BollingerBreakoutStrategy(
            name="Bollinger_Breakout",
            config={"timeframe": timeframes[-1]},
        )

        ensemble = MultiStrategyEnsemble(
            strategies=[rsi_strategy, macd_strategy, bb_strategy],
            min_vote_fraction=0.55,
            min_strategies_agree=2,
            name="ApexEnsemble",
        )

        self.strategy_engine = StrategyEngine()
        self.strategy_engine.register(ensemble)
        self.strategy_engine.add_signal_handler(self.risk_engine.evaluate)

        # ── 5. DerivClient ────────────────────────────────────────────────────
        self.deriv_client = DerivClient(self.config)

        # ── 6. CandleBuilder ─────────────────────────────────────────────────
        self.candle_builder = CandleBuilder(
            timeframes=timeframes,
            valid_timeframes=timeframes,
        )
        # Register candle callback to strategy engine
        self.candle_builder.register_candle_callback(self._on_candle_completed)

        # ── 7. TickCollector ──────────────────────────────────────────────────
        symbols = self.config.get_list("market", "symbols")
        stats_interval = self.config.get_int("market", "stats_interval")

        self.tick_collector = TickCollector(
            provider=self.deriv_client,
            repository=self.repository,
            candle_builder=self.candle_builder,
            symbols=symbols,
            stats_interval=stats_interval,
            valid_timeframes=timeframes,
        )

        # ── 8. Dashboard Server ───────────────────────────────────────────────
        self.dashboard = DashboardServer(self)
        await self.dashboard.start()

        # ── 9. Connect & Start ────────────────────────────────────────────────
        await self.deriv_client.connect()
        await self.tick_collector.start()

        logger.success("Application initialized successfully.")
        logger.info(f"Monitoring symbols: {symbols}")
        logger.info(f"Timeframes: {timeframes}s")

    async def _on_candle_completed(self, candle: Candle) -> None:
        """Callback: forward completed candles to the strategy engine."""
        # Update ATR in portfolio for risk sizing (using BB width as proxy if no ATR)
        await self.strategy_engine.on_candle(candle)

        # Check stop/target hits on every candle close
        if self.broker is not None:
            await self.broker.check_stops_and_targets(
                symbol=candle.symbol,
                current_price=candle.close,
                current_time_ms=candle.timestamp,
            )

        # Update portfolio price
        if self.portfolio is not None:
            self.portfolio.update_price(candle.symbol, candle.close)

    async def _on_order_approved(self, order: TradeOrder) -> None:
        """Callback: RiskEngine approved an order → send to broker."""
        if self.broker is None or self.portfolio is None:
            return

        position = await self.broker.open_position(order)
        self.portfolio.on_position_opened(position)

        # Log portfolio snapshot
        summary = self.portfolio.summary()
        logger.info(
            f"[Portfolio] Equity=${summary['equity']:,.2f} "
            f"Positions={summary['open_positions']} "
            f"Daily={summary['daily_pnl_pct']:+.2f}% "
            f"Drawdown={summary['drawdown_pct']:.2f}%"
        )

    async def _on_trade_closed(self, trade: Trade) -> None:
        """Callback: broker closed a position → update portfolio."""
        if self.portfolio is not None:
            self.portfolio.on_trade_closed(trade)

    async def shutdown(self) -> None:
        """Cleanly stop all services."""
        logger.info("Shutting down services...")

        if self.dashboard is not None:
            await self.dashboard.stop()

        if self.tick_collector is not None:
            await self.tick_collector.stop()

        if self.deriv_client is not None:
            await self.deriv_client.disconnect()

        if self.portfolio is not None:
            summary = self.portfolio.summary()
            logger.info(
                f"Final Portfolio | "
                f"Equity=${summary['equity']:,.2f} "
                f"Return={summary['total_return_pct']:+.2f}% "
                f"Trades={summary['closed_trades']} "
                f"WinRate={summary['win_rate_pct']:.1f}% "
                f"MaxDD={summary['drawdown_pct']:.2f}%"
            )

        if self.database is not None:
            self.database.close()

        logger.success("Application stopped successfully.")

    async def run(self) -> None:
        """Run the application until interrupted."""
        try:
            await self.initialize()
            # Keep application running — log strategy stats every hour
            while True:
                await asyncio.sleep(3600)
                if self.strategy_engine:
                    stats = self.strategy_engine.get_stats()
                    logger.info(f"[StrategyEngine] Stats: {stats}")
                if self.portfolio:
                    logger.info(f"[Portfolio] {self.portfolio.summary()}")
        finally:
            await self.shutdown()