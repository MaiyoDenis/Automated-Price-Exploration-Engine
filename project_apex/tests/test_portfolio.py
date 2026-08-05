"""
Project APEX — Tests for Portfolio State Tracker

Covers:
- Initial capital & cash balance
- Double-entry bookkeeping on position open (stake deduction)
- Double-entry bookkeeping on trade close (stake + PnL return)
- Equity updates with unrealized PnL from live price ticks
- Drawdown percentage tracking against peak equity
- Daily PnL tracking and day-change reset logic
- Summary report dictionary structure
- RiskEngine getters integration
"""
from __future__ import annotations

import time
import pytest

from project_apex.execution.models import Position, Trade
from project_apex.execution.portfolio import Portfolio
from project_apex.risk.models import Direction


def _make_pos(
    symbol: str = "R_25",
    direction: Direction = Direction.LONG,
    size: float = 2.0,
    entry_price: float = 100.0,
) -> Position:
    return Position(
        symbol=symbol,
        direction=direction,
        size=size,
        entry_price=entry_price,
        stop_loss=95.0,
        take_profit=110.0,
        opened_at=int(time.time() * 1000),
    )


class TestPortfolio:
    def test_initial_state(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        assert p.equity == 10_000.0
        assert p.open_position_count == 0
        assert p.daily_pnl_pct == 0.0
        assert p.drawdown_pct == 0.0

    def test_on_position_opened_deducts_stake_from_cash(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        pos = _make_pos(size=2.0, entry_price=100.0)  # stake = 200.0

        p.on_position_opened(pos)

        assert p.open_position_count == 1
        # cash = 10,000 - 200 = 9,800
        assert p._cash == pytest.approx(9_800.0)
        # Without price update, unrealized PnL is 0 -> equity remains 10,000
        assert p.equity == pytest.approx(10_000.0)

    def test_equity_includes_unrealized_pnl(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        pos = _make_pos(size=2.0, entry_price=100.0)
        p.on_position_opened(pos)

        # Price rises to 105.0 -> unrealized PnL = (105 - 100) * 2 = 10.0
        p.update_price("R_25", 105.0)

        assert p.equity == pytest.approx(10_010.0)

    def test_on_trade_closed_returns_stake_plus_pnl(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        pos = _make_pos(size=2.0, entry_price=100.0)
        p.on_position_opened(pos)

        trade = Trade(
            id=pos.id,
            symbol="R_25",
            direction=Direction.LONG,
            size=2.0,
            entry_price=100.0,
            exit_price=110.0,
            opened_at=pos.opened_at,
            closed_at=pos.opened_at + 1000,
            realized_pnl=20.0,
            realized_pnl_pct=0.10,
            close_reason="manual",
            strategy_name="test",
        )

        p.on_trade_closed(trade)

        assert p.open_position_count == 0
        # Returned = 200 stake + 20 PnL = 220 -> cash = 9,800 + 220 = 10,020
        assert p._cash == pytest.approx(10_020.0)
        assert p.equity == pytest.approx(10_020.0)
        assert p.daily_pnl_pct == pytest.approx(20.0 / 10_000.0)

    def test_drawdown_calculation_from_peak(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        
        # 1. Profitable trade pushes peak to 12,000
        pos1 = _make_pos(symbol="R_25", size=1.0, entry_price=100.0)
        pos1.id = "1"
        p.on_position_opened(pos1)
        
        trade1 = Trade("1", "R_25", Direction.LONG, 1.0, 100.0, 200.0, 0, 0, 2000.0, 2.0, "tp", "s")
        p.on_trade_closed(trade1)
        assert p._peak_equity == pytest.approx(12_000.0)
        assert p.drawdown_pct == 0.0

        # 2. Losing trade drops equity to 9,600
        pos2 = _make_pos(symbol="R_25", size=1.0, entry_price=100.0)
        pos2.id = "2"
        p.on_position_opened(pos2)
        
        trade2 = Trade("2", "R_25", Direction.LONG, 1.0, 100.0, 0.0, 0, 0, -2400.0, -0.24, "sl", "s")
        p.on_trade_closed(trade2)

        # Drawdown = (12,000 - 9,600) / 12,000 = 2,400 / 12,000 = 0.20 (20%)
        assert p.drawdown_pct == pytest.approx(0.20)

    def test_daily_pnl_reset_on_day_change(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        trade = Trade("1", "R_25", Direction.LONG, 1.0, 100.0, 110.0, 0, 0, 100.0, 0.1, "tp", "s")
        p.on_trade_closed(trade)

        assert p._daily_realized_pnl == pytest.approx(100.0)

        # Simulate day change
        p._daily_reset_day -= 1
        trade2 = Trade("2", "R_25", Direction.LONG, 1.0, 100.0, 105.0, 0, 0, 50.0, 0.05, "tp", "s")
        p.on_trade_closed(trade2)

        # Daily PnL should reset and only contain trade2's PnL
        assert p._daily_realized_pnl == pytest.approx(50.0)

    def test_summary_dictionary_fields(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        s = p.summary()

        expected_keys = {
            "initial_capital", "cash", "equity", "total_return_pct",
            "daily_pnl_pct", "drawdown_pct", "peak_equity", "open_positions",
            "closed_trades", "win_rate_pct", "total_realized_pnl"
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_get_risk_getters_returns_working_callables(self) -> None:
        p = Portfolio(initial_capital=10_000.0)
        p.update_atr("R_25", 1.5)
        getters = p.get_risk_getters()

        assert getters["equity"]() == pytest.approx(10_000.0)
        assert getters["open_position_count"]() == 0
        assert getters["daily_pnl_pct"]() == 0.0
        assert getters["drawdown_pct"]() == 0.0
        assert getters["atr"]("R_25") == pytest.approx(1.5)
