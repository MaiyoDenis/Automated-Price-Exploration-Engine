"""
Tests for Paper Broker.
"""
import pytest
import asyncio
from project_apex.execution.paper_broker import PaperBroker
from project_apex.risk.models import TradeOrder, Direction


@pytest.mark.asyncio
async def test_paper_broker_open_close():
    broker = PaperBroker(slippage_pct=0.001, spread_pct=0.002)
    
    # 1. Open Long
    order = TradeOrder(
        symbol="R_25",
        direction=Direction.LONG,
        size=10.0,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        strategy_name="Test",
        signal_confidence=1.0
    )
    
    pos = await broker.open_position(order)
    assert pos is not None
    assert pos.id in broker.open_positions
    # price + spread(0.001) + slippage(0.001) = 100.2
    assert pos.entry_price == pytest.approx(100.2)
    
    # 2. Close Long at profit
    trade = await broker.close_position(pos.id, exit_price=110.0)
    assert trade is not None
    assert pos.id not in broker.open_positions
    
    # exit price: 110 - spread(0.001) - slippage(0.001) = 110 - 0.11 - 0.11 = 109.78
    assert trade.exit_price == pytest.approx(109.78)
    assert trade.realized_pnl > 0


@pytest.mark.asyncio
async def test_paper_broker_stops():
    broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)
    order = TradeOrder(
        symbol="R_25",
        direction=Direction.LONG,
        size=10.0,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        strategy_name="Test",
        signal_confidence=1.0
    )
    
    pos = await broker.open_position(order)
    
    # Price moves to 95 -> no stop hit
    closed = await broker.check_stops_and_targets("R_25", 95.0)
    assert len(closed) == 0
    assert broker.open_position_count == 1
    
    # Price moves to 89 -> stop hit
    closed = await broker.check_stops_and_targets("R_25", 89.0)
    assert len(closed) == 1
    assert broker.open_position_count == 0
    assert closed[0].close_reason == "stop_loss"
    assert closed[0].realized_pnl < 0
