"""
Project APEX — Tests for Risk Engine

Covers:
- Rejection of non-entry signals (HOLD, CLOSE_LONG, CLOSE_SHORT)
- Trading halt & resume functionality
- Minimum confidence filtering
- Daily loss limit breach & auto-halt
- Max drawdown limit breach & auto-halt
- Maximum open positions limit
- Symbol cooldown period enforcement
- Position sizing via fixed fractional risk
- ATR-based stop-loss & take-profit calculation (LONG and SHORT)
- Fallback percentage-based stop/TP when ATR unavailable
- Async order handler dispatching
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock
import pytest

from project_apex.risk.engine import RiskEngine
from project_apex.risk.models import Direction, RiskDecision, TradeOrder
from project_apex.strategies.signals import SignalType, TradeSignal


def _make_signal(
    symbol: str = "R_25",
    signal_type: SignalType = SignalType.BUY,
    confidence: float = 0.8,
    price: float = 100.0,
    timestamp: int = 1_000_000,
    strategy_name: str = "test_strat",
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=confidence,
        price=price,
        timestamp=timestamp,
        strategy_name=strategy_name,
    )


class TestRiskEngineRules:
    def test_non_entry_signals_rejected(self) -> None:
        engine = RiskEngine()
        for st in (SignalType.HOLD, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT):
            sig = _make_signal(signal_type=st)
            decision = engine._evaluate_sync(sig)
            assert not decision.approved
            assert "Non-entry signal type" in decision.reason

    def test_trading_halt_rejects_signals(self) -> None:
        engine = RiskEngine()
        engine.halt_trading("Emergency stop")
        sig = _make_signal()
        decision = engine._evaluate_sync(sig)
        assert not decision.approved
        assert "Trading halted: Emergency stop" in decision.reason

    def test_resume_trading_clears_halt(self) -> None:
        engine = RiskEngine()
        engine.halt_trading("Temporary error")
        engine.resume_trading()
        sig = _make_signal()
        decision = engine._evaluate_sync(sig)
        assert decision.approved

    def test_min_confidence_rejection(self) -> None:
        engine = RiskEngine(min_confidence=0.5)
        sig = _make_signal(confidence=0.4)
        decision = engine._evaluate_sync(sig)
        assert not decision.approved
        assert "Confidence 0.40 below minimum 0.50" in decision.reason

    def test_daily_loss_limit_breached_halts_trading(self) -> None:
        engine = RiskEngine(max_daily_loss_pct=0.05)
        # Inject portfolio getter returning -6% loss
        engine.set_portfolio_getters(
            daily_pnl_pct=lambda: -0.06,
            drawdown_pct=lambda: 0.0,
            open_position_count=lambda: 0,
            equity=lambda: 10_000.0,
            atr=lambda _: None,
        )
        sig = _make_signal()
        decision = engine._evaluate_sync(sig)
        assert not decision.approved
        assert engine._trading_halt
        assert "Daily loss limit" in decision.reason

    def test_max_drawdown_breached_halts_trading(self) -> None:
        engine = RiskEngine(max_drawdown_pct=0.15)
        # Inject portfolio getter returning 20% drawdown
        engine.set_portfolio_getters(
            daily_pnl_pct=lambda: 0.0,
            drawdown_pct=lambda: 0.20,
            open_position_count=lambda: 0,
            equity=lambda: 10_000.0,
            atr=lambda _: None,
        )
        sig = _make_signal()
        decision = engine._evaluate_sync(sig)
        assert not decision.approved
        assert engine._trading_halt
        assert "Max drawdown" in decision.reason

    def test_max_open_positions_rejection(self) -> None:
        engine = RiskEngine(max_open_positions=2)
        engine.set_portfolio_getters(
            daily_pnl_pct=lambda: 0.0,
            drawdown_pct=lambda: 0.0,
            open_position_count=lambda: 2,
            equity=lambda: 10_000.0,
            atr=lambda _: None,
        )
        sig = _make_signal()
        decision = engine._evaluate_sync(sig)
        assert not decision.approved
        assert "Max open positions reached" in decision.reason

    def test_symbol_cooldown_rejection(self) -> None:
        engine = RiskEngine(symbol_cooldown_s=60.0)
        sig = _make_signal(symbol="R_25")
        
        # First signal passes
        dec1 = engine._evaluate_sync(sig)
        assert dec1.approved

        # Second signal immediately after fails due to cooldown
        dec2 = engine._evaluate_sync(sig)
        assert not dec2.approved
        assert "Symbol cooldown: R_25 traded" in dec2.reason

    def test_atr_stop_loss_and_take_profit_levels_long(self) -> None:
        engine = RiskEngine(
            risk_per_trade_pct=0.01,
            atr_stop_multiplier=2.0,
            atr_tp_multiplier=4.0,
        )
        engine.set_portfolio_getters(
            daily_pnl_pct=lambda: 0.0,
            drawdown_pct=lambda: 0.0,
            open_position_count=lambda: 0,
            equity=lambda: 10_000.0,
            atr=lambda sym: 2.0 if sym == "R_25" else None,
        )
        sig = _make_signal(symbol="R_25", signal_type=SignalType.BUY, price=100.0)
        decision = engine._evaluate_sync(sig)

        assert decision.approved
        order = decision.order
        assert order is not None
        assert order.direction == Direction.LONG
        # stop_distance = 2.0 * 2.0 = 4.0 -> SL = 100 - 4 = 96.0
        assert order.stop_loss == pytest.approx(96.0)
        # tp_distance = 2.0 * 4.0 = 8.0 -> TP = 100 + 8 = 108.0
        assert order.take_profit == pytest.approx(108.0)
        # risk_amount = 10,000 * 0.01 = 100. Size = 100 / 4.0 = 25.0
        assert order.size == pytest.approx(25.0)

    def test_atr_stop_loss_and_take_profit_levels_short(self) -> None:
        engine = RiskEngine(
            risk_per_trade_pct=0.01,
            atr_stop_multiplier=1.5,
            atr_tp_multiplier=3.0,
        )
        engine.set_portfolio_getters(
            daily_pnl_pct=lambda: 0.0,
            drawdown_pct=lambda: 0.0,
            open_position_count=lambda: 0,
            equity=lambda: 10_000.0,
            atr=lambda sym: 2.0 if sym == "R_25" else None,
        )
        sig = _make_signal(symbol="R_25", signal_type=SignalType.SELL, price=100.0)
        decision = engine._evaluate_sync(sig)

        assert decision.approved
        order = decision.order
        assert order is not None
        assert order.direction == Direction.SHORT
        # stop_distance = 2.0 * 1.5 = 3.0 -> SL = 100 + 3 = 103.0
        assert order.stop_loss == pytest.approx(103.0)
        # tp_distance = 2.0 * 3.0 = 6.0 -> TP = 100 - 6 = 94.0
        assert order.take_profit == pytest.approx(94.0)

    def test_fallback_stop_loss_when_no_atr(self) -> None:
        engine = RiskEngine()
        sig = _make_signal(price=100.0)
        decision = engine._evaluate_sync(sig)

        assert decision.approved
        order = decision.order
        assert order is not None
        # Fallback stop_distance = 1% of 100 = 1.0 -> SL = 99.0
        assert order.stop_loss == pytest.approx(99.0)
        # Fallback tp_distance = 2% of 100 = 2.0 -> TP = 102.0
        assert order.take_profit == pytest.approx(102.0)

    @pytest.mark.asyncio
    async def test_async_order_handler_called_on_approval(self) -> None:
        engine = RiskEngine()
        handler_mock = AsyncMock()
        engine.add_order_handler(handler_mock)

        sig = _make_signal()
        await engine.evaluate(sig)

        handler_mock.assert_called_once()
        order_passed = handler_mock.call_args[0][0]
        assert isinstance(order_passed, TradeOrder)
        assert order_passed.symbol == "R_25"
