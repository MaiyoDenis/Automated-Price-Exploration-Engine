"""
Tests for Risk Engine.
"""
import pytest
from project_apex.risk.engine import RiskEngine
from project_apex.risk.models import TradeOrder, Direction
from project_apex.strategies.signals import TradeSignal, SignalType


@pytest.fixture
def risk_engine():
    engine = RiskEngine(
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,
        max_open_positions=3,
        symbol_cooldown_s=60.0,
        risk_per_trade_pct=0.01,
        min_confidence=0.5
    )
    
    # Mock getters
    engine.set_portfolio_getters(
        daily_pnl_pct=lambda: 0.0,
        drawdown_pct=lambda: 0.0,
        open_position_count=lambda: 0,
        equity=lambda: 10000.0,
        atr=lambda sym: 1.0  # 1.0 ATR
    )
    return engine


def test_risk_approve_valid_signal(risk_engine):
    signal = TradeSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.8,
        price=100.0,
        timestamp=10000,
        strategy_name="Test"
    )
    decision = risk_engine._evaluate_sync(signal)
    assert decision.approved is True
    assert decision.order is not None
    assert decision.order.direction == Direction.LONG
    
    # Risk per trade = 1% of 10000 = 100
    # ATR = 1.0, stop distance = 1.5 * 1.0 = 1.5
    # size = 100 / 1.5 = 66.6667
    assert decision.order.size == pytest.approx(66.6667, abs=0.01)


def test_risk_reject_low_confidence(risk_engine):
    signal = TradeSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.2, # < 0.5 minimum
        price=100.0,
        timestamp=10000,
        strategy_name="Test"
    )
    decision = risk_engine._evaluate_sync(signal)
    assert decision.approved is False
    assert "Confidence" in decision.reason


def test_risk_reject_max_positions(risk_engine):
    risk_engine._get_open_position_count = lambda: 3
    signal = TradeSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.8,
        price=100.0,
        timestamp=10000,
        strategy_name="Test"
    )
    decision = risk_engine._evaluate_sync(signal)
    assert decision.approved is False
    assert "Max open positions" in decision.reason


def test_risk_reject_drawdown(risk_engine):
    risk_engine._get_drawdown_pct = lambda: 0.20 # > 0.15 max
    signal = TradeSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.8,
        price=100.0,
        timestamp=10000,
        strategy_name="Test"
    )
    decision = risk_engine._evaluate_sync(signal)
    assert decision.approved is False
    assert "drawdown" in decision.reason
    assert risk_engine._trading_halt is True
