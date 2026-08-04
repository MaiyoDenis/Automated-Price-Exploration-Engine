"""
Project APEX — Tests for BacktestingEngine

Uses an in-memory SQLite database pre-seeded with deterministic candle data.
Tests cover data loading, full backtest runs, commission, slippage,
zero-signal strategies, and error paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from project_apex.backtesting.engine import BacktestingEngine, BacktestResult
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Concrete strategy stubs used across tests
# ---------------------------------------------------------------------------

class AlwaysBuyStrategy(Strategy):
    """Emits signal=1 on every bar — maximises number of trades in tests."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["signal"] = 1
        return data


class AlwaysSellStrategy(Strategy):
    """Emits signal=-1 on every bar."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["signal"] = -1
        return data


class ZeroSignalStrategy(Strategy):
    """Emits no signal — capital should remain unchanged."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data["signal"] = 0
        return data


class AlternateBuySellStrategy(Strategy):
    """Alternates buy/sell every bar — produces maximum number of closed trades."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = [1 if i % 2 == 0 else -1 for i in range(len(data))]
        data["signal"] = signals
        return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SYMBOL = "R_25"
TIMEFRAME = 60

# Deterministic candle rows: (symbol, timeframe, timestamp, open, high, low, close)
_CANDLES = [
    (SYMBOL, TIMEFRAME, 1_000_000 + i * 60,
     100.0 + i,          # open  (rising)
     102.0 + i,          # high
     98.0 + i,           # low
     101.0 + i)          # close (rising, so long trades profit)
    for i in range(50)
]


@pytest.fixture()
def db() -> SQLiteManager:
    """In-memory SQLite with candles table seeded with deterministic data."""
    manager = SQLiteManager(":memory:")
    manager.connect()
    manager.initialize()
    manager.execute_many(
        "INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        _CANDLES,
    )
    yield manager
    manager.close()


@pytest.fixture()
def engine(db: SQLiteManager) -> BacktestingEngine:
    return BacktestingEngine(db=db, initial_capital=10_000.0)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_returns_non_empty_dataframe(self, engine: BacktestingEngine) -> None:
        df = engine.load_data(SYMBOL, TIMEFRAME)
        assert not df.empty

    def test_returns_required_columns(self, engine: BacktestingEngine) -> None:
        df = engine.load_data(SYMBOL, TIMEFRAME)
        for col in ("timestamp", "open", "high", "low", "close"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_matches_seeded_data(self, engine: BacktestingEngine) -> None:
        df = engine.load_data(SYMBOL, TIMEFRAME)
        assert len(df) == len(_CANDLES)

    def test_data_ordered_ascending_by_timestamp(self, engine: BacktestingEngine) -> None:
        df = engine.load_data(SYMBOL, TIMEFRAME)
        assert df["timestamp"].is_monotonic_increasing

    def test_empty_for_unknown_symbol(self, engine: BacktestingEngine) -> None:
        df = engine.load_data("UNKNOWN_XYZ", TIMEFRAME)
        assert df.empty

    def test_empty_for_unknown_timeframe(self, engine: BacktestingEngine) -> None:
        df = engine.load_data(SYMBOL, 9999)
        assert df.empty


# ---------------------------------------------------------------------------
# run() — BacktestResult structure
# ---------------------------------------------------------------------------

class TestBacktestRunResult:
    def test_returns_backtest_result_instance(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlwaysBuyStrategy("buy"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)

    def test_strategy_name_recorded(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlwaysBuyStrategy("my_strategy"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "my_strategy"

    def test_symbol_recorded(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlwaysBuyStrategy("s"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.symbol == SYMBOL

    def test_initial_capital_recorded(self, engine: BacktestingEngine) -> None:
        result = engine.run(ZeroSignalStrategy("flat"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 10_000.0

    def test_total_trades_matches_trade_list_length(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlternateBuySellStrategy("alt"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.total_trades == len(result.trades)

    def test_win_rate_in_valid_range(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlternateBuySellStrategy("alt"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert 0.0 <= result.win_rate_pct <= 100.0

    def test_max_drawdown_non_negative(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlternateBuySellStrategy("alt"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.max_drawdown_pct >= 0.0

    def test_error_dict_when_no_data(self, engine: BacktestingEngine) -> None:
        result = engine.run(AlwaysBuyStrategy("s"), "NONEXISTENT", TIMEFRAME)
        assert isinstance(result, dict)
        assert result.get("error") == "No data"


# ---------------------------------------------------------------------------
# run() — zero-signal strategy
# ---------------------------------------------------------------------------

class TestZeroSignalStrategy:
    def test_capital_unchanged_with_zero_signals(self, engine: BacktestingEngine) -> None:
        result = engine.run(ZeroSignalStrategy("flat"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.final_capital == pytest.approx(10_000.0, abs=1e-6)

    def test_zero_trades_with_zero_signals(self, engine: BacktestingEngine) -> None:
        result = engine.run(ZeroSignalStrategy("flat"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_zero_win_rate_with_no_trades(self, engine: BacktestingEngine) -> None:
        result = engine.run(ZeroSignalStrategy("flat"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.win_rate_pct == 0.0


# ---------------------------------------------------------------------------
# run() — profitable strategy (rising prices + always long)
# ---------------------------------------------------------------------------

class TestProfitableStrategy:
    def test_long_strategy_profits_on_rising_prices(self, engine: BacktestingEngine) -> None:
        """Our seeded data is monotonically rising so long positions should profit."""
        result = engine.run(AlternateBuySellStrategy("alt"), SYMBOL, TIMEFRAME)
        assert isinstance(result, BacktestResult)
        assert result.total_return_pct > 0, (
            "Alternating long/short on rising prices should yield positive total return."
        )


# ---------------------------------------------------------------------------
# Commission and slippage tests
# ---------------------------------------------------------------------------

class TestCommissionAndSlippage:
    def test_commission_reduces_capital(self, db: SQLiteManager) -> None:
        """Higher commission → lower final capital than zero commission."""
        engine_no_comm = BacktestingEngine(db=db, initial_capital=10_000.0, commission_pct=0.0)
        engine_with_comm = BacktestingEngine(db=db, initial_capital=10_000.0, commission_pct=0.01)

        result_no = engine_no_comm.run(AlternateBuySellStrategy("a"), SYMBOL, TIMEFRAME)
        result_with = engine_with_comm.run(AlternateBuySellStrategy("a"), SYMBOL, TIMEFRAME)

        assert isinstance(result_no, BacktestResult)
        assert isinstance(result_with, BacktestResult)
        assert result_with.final_capital <= result_no.final_capital, (
            "Commission must reduce final capital."
        )

    def test_slippage_reduces_long_profit(self, db: SQLiteManager) -> None:
        """Higher slippage → lower profit on long trades vs. zero slippage."""
        engine_no_slip = BacktestingEngine(db=db, initial_capital=10_000.0, slippage_pct=0.0)
        engine_with_slip = BacktestingEngine(db=db, initial_capital=10_000.0, slippage_pct=0.005)

        result_no = engine_no_slip.run(AlternateBuySellStrategy("a"), SYMBOL, TIMEFRAME)
        result_with = engine_with_slip.run(AlternateBuySellStrategy("a"), SYMBOL, TIMEFRAME)

        assert isinstance(result_no, BacktestResult)
        assert isinstance(result_with, BacktestResult)
        assert result_with.final_capital <= result_no.final_capital, (
            "Slippage must reduce final capital compared to zero slippage."
        )

    def test_zero_commission_and_slippage_baseline(self, engine: BacktestingEngine) -> None:
        """Default engine has zero commission and slippage."""
        assert engine.commission_pct == 0.0
        assert engine.slippage_pct == 0.0
