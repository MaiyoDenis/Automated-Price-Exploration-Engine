"""
Project APEX
SQLite Market Data Repository
"""

from __future__ import annotations

from project_apex.database.ports import MarketDataRepository
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


class SQLiteMarketDataRepository(MarketDataRepository):
    """SQLite implementation of the MarketDataRepository port."""

    def __init__(self, manager: SQLiteManager) -> None:
        self._manager = manager

    def initialize(self) -> None:
        """Create required tables and unique indexes."""
        self._manager.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL
            )
            """
        )
        self._manager.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ticks_symbol_timestamp
            ON ticks (symbol, timestamp)
            """
        )

        self._manager.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL
            )
            """
        )
        self._manager.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_timestamp
            ON candles (symbol, timeframe, timestamp)
            """
        )

    def save_tick(self, tick: Tick) -> None:
        self._manager.execute(
            """
            INSERT OR IGNORE INTO ticks (symbol, timestamp, price)
            VALUES (?, ?, ?)
            """,
            (tick.symbol, tick.timestamp, tick.price)
        )

    def save_candle(self, candle: Candle) -> None:
        self._manager.execute(
            """
            INSERT OR IGNORE INTO candles (symbol, timeframe, timestamp, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (candle.symbol, candle.timeframe, candle.timestamp, candle.open, candle.high, candle.low, candle.close)
        )

    def get_latest_tick_timestamp(self, symbol: str) -> int | None:
        rows = self._manager.fetchall(
            """
            SELECT MAX(timestamp) FROM ticks WHERE symbol = ?
            """,
            (symbol,)
        )
        if rows and rows[0][0] is not None:
            return int(rows[0][0])
        return None

    def get_ticks(self, symbol: str, start: int, end: int) -> list[Tick]:
        rows = self._manager.fetchall(
            """
            SELECT symbol, timestamp, price
            FROM ticks
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """,
            (symbol, start, end)
        )
        return [
            Tick(
                symbol=row[0],
                timestamp=int(row[1]),
                price=float(row[2])
            )
            for row in rows
        ]

    def get_candles(self, symbol: str, timeframe: int, start: int, end: int) -> list[Candle]:
        rows = self._manager.fetchall(
            """
            SELECT symbol, timeframe, timestamp, open, high, low, close
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """,
            (symbol, timeframe, start, end)
        )
        return [
            Candle(
                symbol=row[0],
                timeframe=int(row[1]),
                timestamp=int(row[2]),
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                tick_count=0
            )
            for row in rows
        ]
