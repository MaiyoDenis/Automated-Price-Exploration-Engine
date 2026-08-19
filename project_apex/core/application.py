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
from project_apex.strategies.scalping import VWAPScalperStrategy
from project_apex.strategies.differs_strategy import DiffersStrategy

# Risk layer
from project_apex.risk.engine import RiskEngine
from project_apex.risk.circuit_breaker import CircuitBreaker
from project_apex.risk.differs_risk_engine import DiffersRiskEngine

# Execution layer
from project_apex.execution.paper_broker import PaperBroker
from project_apex.execution.live_broker import LiveBroker
from project_apex.execution.portfolio import Portfolio
from project_apex.execution.models import Trade
from project_apex.risk.models import TradeOrder
from project_apex.execution.differs_executor import DiffersExecutor

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

    def __init__(self) -> None:
        setup_logger()
        logger.info("Starting Project APEX — Elite Autonomous Mode...")

        self.config = Config()
        # Read paper_trading from config so that config.yaml changes take effect on restart.
        self.paper_trading: bool = bool(self.config.get("trading", "paper_trading") if self.config.get("trading", "paper_trading") is not None else True)
        logger.info(f"Mode: {'Paper Trading' if self.paper_trading else 'LIVE Trading'}")
        # Validate credentials eagerly — fail at startup with a clear message
        # rather than deep inside connect() with a cryptic network error.
        self.environment = Environment(require_live_credentials=not self.paper_trading)

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
        
        self.trading_mode: str = "standard"
        self.standard_risk_engine: Optional[RiskEngine] = None
        self.scalping_risk_engine: Optional[RiskEngine] = None
        self.standard_strategy_engine: Optional[StrategyEngine] = None
        self.scalping_strategy_engine: Optional[StrategyEngine] = None
        
        # Differs Mode
        self.differs_risk_engine: Optional[DiffersRiskEngine] = None
        self.differs_strategy_engine: Optional[StrategyEngine] = None
        self.differs_executor: Optional[DiffersExecutor] = None

    async def initialize(self) -> None:
        """Start all required services in dependency order."""
        logger.info("Initializing services...")

        # ── 1. Database & Repository ──────────────────────────────────────────
        db_path = self.config.get_str("database", "path")
        self.database = SQLiteManager(db_path)
        self.database.connect()
        self.database.initialize()
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
            # deriv_client is wired in step 8; LiveBroker holds a reference to it.
            # We create deriv_client early so LiveBroker can be initialised here.
            self.deriv_client = DerivClient(self.config)
            self.broker = LiveBroker(client=self.deriv_client)

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

        # ── 5. Standard Risk Engine ───────────────────────────────────────────
        self.standard_risk_engine = RiskEngine(
            max_daily_loss_pct=float(self.config.get("risk", "max_daily_loss_pct") or 0.05),
            max_drawdown_pct=float(self.config.get("risk", "max_drawdown_pct") or 0.15),
            max_open_positions=int(self.config.get("risk", "max_open_positions") or 3),
            symbol_cooldown_s=float(self.config.get("risk", "symbol_cooldown_s") or 60.0),
            risk_per_trade_pct=float(self.config.get("risk", "risk_per_trade_pct") or 0.01),
            kelly_fraction=float(self.config.get("risk", "kelly_fraction") or 0.25),
            min_confidence=float(self.config.get("risk", "min_confidence") or 0.3),
        )
        self.standard_risk_engine.set_circuit_breaker(self.circuit_breaker)
        getters = self.portfolio.get_risk_getters()
        self.standard_risk_engine.set_portfolio_getters(**getters)
        self.standard_risk_engine.add_order_handler(self._on_order_approved)

        # ── 5b. Scalping Risk Engine ──────────────────────────────────────────
        self.scalping_risk_engine = RiskEngine(
            max_daily_loss_pct=float(self.config.get("risk", "max_daily_loss_pct") or 0.05),
            max_drawdown_pct=float(self.config.get("risk", "max_drawdown_pct") or 0.15),
            max_open_positions=int(self.config.get("risk", "max_open_positions") or 5), # Allow more for scalping
            symbol_cooldown_s=10.0, # Tight cooldown
            risk_per_trade_pct=float(self.config.get("risk", "risk_per_trade_pct") or 0.01),
            kelly_fraction=0.1, # Less Kelly risk per trade for scalping
            atr_stop_multiplier=0.5, # Very tight SL
            atr_tp_multiplier=1.0,   # Quick TP
            min_confidence=0.8,
            enable_correlation_filter=False, # Speed over correlation
        )
        self.scalping_risk_engine.set_circuit_breaker(self.circuit_breaker)
        self.scalping_risk_engine.set_portfolio_getters(**getters)
        self.scalping_risk_engine.add_order_handler(self._on_order_approved)

        # ── 6. Standard Strategy Engine (Ensemble) ────────────────────────────
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
            db=self.database,
        )

        ml_strategy = MLStrategy(
            name="ML_XGBoost",
            config={"timeframe": primary_tf, "model_path": "datasets/xgb_model.joblib"},
        )

        meta_strategy = MetaRegimeStrategy(
            trend_strategies=[macd_strategy, ml_strategy],
            ranging_strategies=[rsi_strategy, bb_strategy],
            name="MetaRegimeRouter",
            config={"timeframe": primary_tf},
        )

        self.standard_strategy_engine = StrategyEngine()
        self.standard_strategy_engine.register(meta_strategy)
        self.standard_strategy_engine.register(self.ensemble)
        self.standard_strategy_engine.add_signal_handler(self.standard_risk_engine.evaluate)

        # ── 6b. Scalping Strategy Engine ──────────────────────────────────────
        scalping_strategy = VWAPScalperStrategy(
            name="VWAP_Scalper",
            config={"timeframe": primary_tf},
        )
        self.scalping_strategy_engine = StrategyEngine()
        self.scalping_strategy_engine.register(scalping_strategy)
        self.scalping_strategy_engine.add_signal_handler(self.scalping_risk_engine.evaluate)

        # ── 6c. Differs Engine ───────────────────────────────────────────────
        # Must only be used in REAL mode as PaperBroker doesn't support digit contracts yet.
        # But for logic flow, we init it here.
        if self.deriv_client:
            self.differs_executor = DiffersExecutor(provider=self.deriv_client, portfolio=self.portfolio)

        self.differs_risk_engine = DiffersRiskEngine(
            max_daily_loss_pct=float(self.config.get("risk", "max_daily_loss_pct") or 0.05),
            loss_cooldown_ticks=5,
            base_stake=2.0,                    # Normal trade stake
            recovery_stake=22.0,               # Recovery stake after 1 loss
            recovery_min_confidence=0.70,      # Must be high-confidence to fire $22
            max_consecutive_losses=2,          # Halt session after 2 losses in a row
            payout_ratio=0.097,                # ~9.7% payout
        )

        # Build a live win-rate getter that reads from the DiffersStrategy instance.
        # This is set after differs_strategy is created (see below) via a closure.
        _differs_strategy_ref: list = []   # will hold the DiffersStrategy instance

        def _live_win_rate(strategy_name: str) -> float | None:
            if not _differs_strategy_ref:
                return None
            s = _differs_strategy_ref[0]
            # Average win rate across all tracked symbols
            all_trades = [t for trades in s.recent_trades.values() for t in trades]
            if len(all_trades) < 8:
                return None
            wins = sum(1 for _, won, _ in all_trades if won)
            return wins / len(all_trades)

        # Pass portfolio getters
        self.differs_risk_engine.set_portfolio_getters(
            daily_pnl_pct=self.portfolio.get_risk_getters()["daily_pnl_pct"],
            open_position_count=lambda: 1 if self.differs_executor and self.differs_executor.active_contract_id else 0,
            equity=self.portfolio.get_risk_getters()["equity"],
            strategy_win_rate=_live_win_rate,   # Live Kelly sizing
        )

        if self.differs_executor:
            self.differs_risk_engine.add_order_handler(self.differs_executor.execute_order)
            
            def _on_differs_settled(won: bool, order: TradeOrder | None) -> None:
                self.differs_risk_engine.on_trade_result(won)
                if order and _differs_strategy_ref:
                    _differs_strategy_ref[0].record_trade_outcome(
                        symbol=order.symbol,
                        excluded_digit=order.metadata.get("barrier", 0),
                        won=won,
                        confidence=order.signal_confidence
                    )
                    
            self.differs_executor.set_settlement_callback(_on_differs_settled)

        differs_strategy = DiffersStrategy(config={"base_confidence": 0.70, "duration_ticks": 1})
        _differs_strategy_ref.append(differs_strategy)   # Wire win-rate getter
        self.differs_strategy_engine = StrategyEngine()
        self.differs_strategy_engine.register(differs_strategy)
        self.differs_strategy_engine.add_signal_handler(self.differs_risk_engine.evaluate)

        # Set default active engines
        self.risk_engine = self.standard_risk_engine
        self.strategy_engine = self.standard_strategy_engine

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
        # Already created above when paper_trading=False; skip duplicate instantiation.
        if self.deriv_client is None:
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
            initial_symbols=initial_symbols,  # Seed so strategies work from tick 1
        )

        # ── 11. Health Monitor ────────────────────────────────────────────────
        # Created before TickCollector so its record_tick can be passed as callback.
        self.health_monitor = HealthMonitor(
            app=self,
            stale_tick_threshold_s=30.0,   # YELLOW after 30s without a tick
            stale_tick_critical_s=120.0,   # RED after 2 min (e.g. full disconnect)
        )
        await self.health_monitor.start()

        # ── 12. TickCollector ─────────────────────────────────────────────────
        stats_interval = self.config.get_int("market", "stats_interval")
        self.tick_collector = TickCollector(
            provider=self.deriv_client,
            repository=self.repository,
            candle_builder=self.candle_builder,
            symbols=initial_symbols,
            stats_interval=stats_interval,
            valid_timeframes=timeframes,
            tick_callback=self._on_tick_received,  # freshness tracked per actual tick + tick strategy routing
        )

        # ── 13. Dashboard Server ──────────────────────────────────────────────
        self.dashboard = DashboardServer(self)
        await self.dashboard.start()

        # ── 14. Connect & Start ───────────────────────────────────────────────
        await self.deriv_client.connect()
        if self.differs_executor:
            await self.differs_executor.start()
        await self.tick_collector.start()
        await self.autopilot.start()
        await self.model_trainer.start()

        logger.success("Application initialized successfully — Autonomous mode ACTIVE.")
        logger.info(f"Initial symbols: {initial_symbols}")
        logger.info(f"Timeframes: {timeframes}s")

    def set_trading_mode(self, mode: str) -> None:
        """Hot-swap the active trading mode (standard, scalping, or differs)."""
        if mode not in ("standard", "scalping", "differs"):
            logger.warning(f"Attempted to set unknown trading mode: {mode}")
            return
            
        if self.trading_mode == mode:
            return
            
        self.trading_mode = mode
        if mode == "standard":
            self.risk_engine = self.standard_risk_engine
            self.strategy_engine = self.standard_strategy_engine
            logger.success("[Application] Mode switched to STANDARD (Ensemble + ML)")
        elif mode == "scalping":
            self.risk_engine = self.scalping_risk_engine
            self.strategy_engine = self.scalping_strategy_engine
            logger.success("[Application] Mode switched to SCALPING (VWAP + EMA)")
        elif mode == "differs":
            # Just set the active engines, ticks will naturally flow there via _on_tick_received
            self.risk_engine = self.differs_risk_engine
            self.strategy_engine = self.differs_strategy_engine
            logger.success("[Application] Mode switched to DIFFERS (Adaptive Digit Analyzer)")
            
        if self.alert_manager:
            self.alert_manager.system(f"Trading mode switched to: {mode.upper()}")

    async def switch_account(self, account_type: str) -> dict:
        """
        Hot-swap the active trading account between 'demo' and 'real'.
        Closes all positions, disconnects, and reconnects with new credentials.
        """
        if account_type not in ("demo", "real"):
            return {"success": False, "error": f"Invalid account type: {account_type}"}

        current_type = self.config.get("api", "account_type") or "demo"
        if current_type == account_type:
            return {"success": True, "message": f"Already on {account_type} account"}

        logger.warning(f"[Application] Switching account from {current_type} → {account_type}")

        # 1. Close all open positions safely
        if self.broker and hasattr(self.broker, "open_positions"):
            for pos_id in list(self.broker.open_positions.keys()):
                try:
                    symbol = self.broker.open_positions[pos_id].symbol
                    current_price = self.portfolio._latest_price.get(symbol, 0.0)
                    await self.broker.close_position(pos_id, current_price, reason="account_switch")
                    logger.info(f"[Application] Closed position {pos_id[:8]} before account switch")
                except Exception as e:
                    logger.warning(f"[Application] Could not close position {pos_id[:8]}: {e}")

        # 2. Stop tick collection and unsubscribe
        active_symbols = list(self.autopilot.active_symbols) if self.autopilot else []
        if self.tick_collector:
            await self.tick_collector.stop()

        # 3. Disconnect WebSocket
        if self.deriv_client:
            await self.deriv_client.disconnect()

        # 4. Update the account type in config
        self.config._config["api"]["account_type"] = account_type

        # 5. Reconnect with new credentials
        try:
            await self.deriv_client.connect()
        except Exception as e:
            logger.error(f"[Application] Reconnect failed after account switch: {e}")
            return {"success": False, "error": str(e)}

        # 6. Re-subscribe to symbols and restart tick collection
        if self.tick_collector and active_symbols:
            await self.tick_collector.start()

        # 7. Update LiveBroker reference (it already holds the same DerivClient)
        logger.success(f"[Application] Account switched to {account_type.upper()} ✓")
        if self.alert_manager:
            self.alert_manager.system(f"Account switched to {account_type.upper()}")

        return {"success": True, "account_type": account_type}

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_tick_received(self, tick: Tick) -> None:
        """
        Route ticks to health monitor, stop/TP checker, and active strategy engine.

        Tick-level stop/TP ensures stops fire at real market prices, not just on
        candle closes (which can be 60s apart on volatility indices).
        """
        if self.health_monitor:
            self.health_monitor.record_tick(tick.symbol)

        # Update latest price in portfolio for live unrealized P&L
        if self.portfolio:
            self.portfolio.update_price(tick.symbol, tick.price)

        # Tick-level stop-loss / take-profit for live broker
        # Paper broker uses candle close (fine for simulation).
        # Live broker needs tick-level resolution to avoid slipping through stops.
        if not self.paper_trading and self.broker:
            asyncio.create_task(
                self.broker.check_stops_and_targets(
                    symbol=tick.symbol,
                    current_price=tick.price,
                    current_time_ms=tick.timestamp,
                )
            )

        # Tick-level strategy routing for the Differs engine
        if self.trading_mode == "differs" and self.strategy_engine:
            asyncio.create_task(self.strategy_engine.on_tick(tick))
            if self.differs_risk_engine:
                self.differs_risk_engine.on_tick()

    async def _on_candle_completed(self, candle: Candle) -> None:
        """Candle from the builder → update portfolio ATR, forward to autopilot."""
        # Update portfolio price for unrealized PnL (also updated per-tick in live mode)
        if self.portfolio:
            self.portfolio.update_price(candle.symbol, candle.close)

        # ── ATR update ────────────────────────────────────────────────────────
        # Compute a rolling ATR(14) from the last 30 candles in the repository
        # and push it into the portfolio so RiskEngine stop sizing is accurate.
        if self.portfolio and self.repository:
            try:
                import time as _time
                start_ts = int(_time.time()) - 86400 * 2  # last 2 days is enough for ATR(14)
                recent = self.repository.get_candles(
                    candle.symbol, candle.timeframe, start_ts, int(_time.time())
                )
                if recent and len(recent) >= 2:
                    import pandas as _pd
                    df = _pd.DataFrame(
                        [(c.high, c.low, c.close) for c in recent],
                        columns=["high", "low", "close"],
                    )
                    prev_close = df["close"].shift(1)
                    tr = _pd.concat([
                        df["high"] - df["low"],
                        (df["high"] - prev_close).abs(),
                        (df["low"] - prev_close).abs(),
                    ], axis=1).max(axis=1)
                    atr = float(tr.tail(14).mean())
                    if atr > 0:
                        self.portfolio.update_atr(candle.symbol, atr)
            except Exception as _exc:
                logger.debug(f"[Application] ATR update failed for {candle.symbol}: {_exc}")

        # Update circuit breaker with latest equity
        if self.circuit_breaker and self.portfolio:
            self.circuit_breaker.update_equity(self.portfolio.equity)

        # Route to autopilot (only active symbols proceed)
        if self.autopilot:
            await self.autopilot.on_candle(candle)

        # Paper mode: candle-level stop/TP check (live mode uses tick-level above)
        if self.paper_trading and self.broker:
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

        if self.differs_executor:
            await self.differs_executor.stop()

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