"""
Project APEX — Tests for Paper Broker

Covers:
- Open position fill price calculations with slippage and spread (LONG and SHORT)
- Tracking open positions state
- Close position PnL calculations for LONG and SHORT
- Automatic stop-loss and take-profit detection on tick price updates
- Async trade handler notifications
"""
from __future__ import annotations

from unittest.mock import AsyncMock
import pytest

from project_apex.execution.models import Position, Trade
from project_apex.execution.paper_broker import PaperBroker
from project_apex.risk.models import Direction, TradeOrder


def _make_order(
    symbol: str = "R_25",
    direction: Direction = Direction.LONG,
    size: float = 1.0,
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit: float = 110.0,
    strategy_name: str = "test_strat",
) -> TradeOrder:
    return TradeOrder(
        symbol=symbol,
        direction=direction,
        size=size,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_name=strategy_name,
        signal_confidence=0.9,
    )


class TestPaperBroker:
    @pytest.mark.asyncio
    async def test_open_position_applies_slippage_and_spread_long(self) -> None:
        # slippage = 0.001 (0.1%), spread = 0.002 (0.2%)
        broker = PaperBroker(slippage_pct=0.001, spread_pct=0.002)
        order = _make_order(direction=Direction.LONG, entry_price=100.0)

        position = await broker.open_position(order)

        # Fill price = price + spread/2 + slippage = 100.0 + 0.1 + 0.1 = 100.2
        assert position.entry_price == pytest.approx(100.2)
        assert broker.open_position_count == 1
        assert position.id in broker.open_positions

    @pytest.mark.asyncio
    async def test_open_position_applies_slippage_and_spread_short(self) -> None:
        broker = PaperBroker(slippage_pct=0.001, spread_pct=0.002)
        order = _make_order(direction=Direction.SHORT, entry_price=100.0)

        position = await broker.open_position(order)

        # Fill price = price - spread/2 - slippage = 100.0 - 0.1 - 0.1 = 99.8
        assert position.entry_price == pytest.approx(99.8)

    @pytest.mark.asyncio
    async def test_close_position_realized_pnl_long_profit(self) -> None:
        broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)  # Zero friction for clean PnL test
        order = _make_order(direction=Direction.LONG, entry_price=100.0, size=2.0)
        pos = await broker.open_position(order)

        trade = await broker.close_position(pos.id, exit_price=110.0, reason="manual")

        assert trade is not None
        assert isinstance(trade, Trade)
        assert trade.exit_price == pytest.approx(110.0)
        # PnL = (110 - 100) * 2 = 20.0
        assert trade.realized_pnl == pytest.approx(20.0)
        assert trade.realized_pnl_pct == pytest.approx(0.10)  # 20 / (100 * 2)
        assert broker.open_position_count == 0

    @pytest.mark.asyncio
    async def test_close_position_realized_pnl_short_profit(self) -> None:
        broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)
        order = _make_order(direction=Direction.SHORT, entry_price=100.0, size=2.0)
        pos = await broker.open_position(order)

        trade = await broker.close_position(pos.id, exit_price=90.0, reason="manual")

        assert trade is not None
        # PnL = (100 - 90) * 2 = 20.0
        assert trade.realized_pnl == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_close_non_existent_position_returns_none(self) -> None:
        broker = PaperBroker()
        result = await broker.close_position("invalid_id", exit_price=100.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_check_stops_and_targets_long_stop_loss(self) -> None:
        broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)
        order = _make_order(direction=Direction.LONG, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        await broker.open_position(order)

        # Price ticks to 94.0 -> triggers SL
        closed = await broker.check_stops_and_targets("R_25", current_price=94.0)

        assert len(closed) == 1
        assert closed[0].close_reason == "stop_loss"
        assert broker.open_position_count == 0

    @pytest.mark.asyncio
    async def test_check_stops_and_targets_long_take_profit(self) -> None:
        broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)
        order = _make_order(direction=Direction.LONG, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        await broker.open_position(order)

        # Price ticks to 111.0 -> triggers TP
        closed = await broker.check_stops_and_targets("R_25", current_price=111.0)

        assert len(closed) == 1
        assert closed[0].close_reason == "take_profit"

    @pytest.mark.asyncio
    async def test_check_stops_and_targets_short_stop_loss(self) -> None:
        broker = PaperBroker(slippage_pct=0.0, spread_pct=0.0)
        order = _make_order(direction=Direction.SHORT, entry_price=100.0, stop_loss=105.0, take_profit=90.0)
        await broker.open_position(order)

        # Price rises to 106.0 -> SHORT stop loss hit
        closed = await broker.check_stops_and_targets("R_25", current_price=106.0)

        assert len(closed) == 1
        assert closed[0].close_reason == "stop_loss"

    @pytest.mark.asyncio
    async def test_async_trade_handler_called_on_close(self) -> None:
        broker = PaperBroker()
        handler_mock = AsyncMock()
        broker.add_trade_handler(handler_mock)

        pos = await broker.open_position(_make_order())
        await broker.close_position(pos.id, exit_price=105.0)

        handler_mock.assert_called_once()
        closed_trade = handler_mock.call_args[0][0]
        assert isinstance(closed_trade, Trade)
