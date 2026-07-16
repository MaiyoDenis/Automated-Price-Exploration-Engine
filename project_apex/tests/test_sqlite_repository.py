"""
Test SQLite Repository
"""

import pytest

from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.database.sqlite_market_data import SQLiteMarketDataRepository
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


@pytest.fixture
def repo():
    manager = SQLiteManager(":memory:")
    manager.connect()
    repository = SQLiteMarketDataRepository(manager)
    repository.initialize()
    return repository


def test_save_and_retrieve_tick(repo: SQLiteMarketDataRepository):
    tick = Tick("R_25", 1000, 1.5)
    repo.save_tick(tick)
    
    ticks = repo.get_ticks("R_25", 0, 2000)
    assert len(ticks) == 1
    assert ticks[0] == tick


def test_save_tick_duplicate_ignored(repo: SQLiteMarketDataRepository):
    tick = Tick("R_25", 1000, 1.5)
    repo.save_tick(tick)
    repo.save_tick(tick)
    
    ticks = repo.get_ticks("R_25", 0, 2000)
    assert len(ticks) == 1


def test_save_and_retrieve_candle(repo: SQLiteMarketDataRepository):
    candle = Candle("R_25", 60, 60000, 1.0, 2.0, 0.5, 1.5, 0)
    repo.save_candle(candle)
    
    candles = repo.get_candles("R_25", 60, 0, 120000)
    assert len(candles) == 1
    assert candles[0] == candle


def test_save_candle_duplicate_ignored(repo: SQLiteMarketDataRepository):
    candle = Candle("R_25", 60, 60000, 1.0, 2.0, 0.5, 1.5, 0)
    repo.save_candle(candle)
    repo.save_candle(candle)
    
    candles = repo.get_candles("R_25", 60, 0, 120000)
    assert len(candles) == 1


def test_get_latest_tick_timestamp_returns_max(repo: SQLiteMarketDataRepository):
    repo.save_tick(Tick("R_25", 1000, 1.5))
    repo.save_tick(Tick("R_25", 3000, 2.5))
    repo.save_tick(Tick("R_25", 2000, 2.0))
    
    latest = repo.get_latest_tick_timestamp("R_25")
    assert latest == 3000


def test_get_latest_tick_timestamp_returns_none_when_empty(repo: SQLiteMarketDataRepository):
    latest = repo.get_latest_tick_timestamp("R_25")
    assert latest is None


def test_unique_index_ticks_exists(repo: SQLiteMarketDataRepository):
    rows = repo._manager.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_ticks_symbol_timestamp'"
    )
    assert len(rows) == 1


def test_unique_index_candles_exists(repo: SQLiteMarketDataRepository):
    rows = repo._manager.fetchall(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_candles_symbol_timeframe_timestamp'"
    )
    assert len(rows) == 1
