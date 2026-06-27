"""
Project APEX
Database Manager
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()

    def execute(self, query: str, params: tuple = ()) -> None:
        """Execute INSERT/UPDATE/DELETE queries."""
        if self.connection is None:
            raise RuntimeError("Database is not connected.")

        with self.connection:
            self.connection.execute(query, params)

    def fetchall(self, query: str, params: tuple = ()) -> list:
        """Execute SELECT queries."""
        if self.connection is None:
            raise RuntimeError("Database is not connected.")

        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def initialize(self) -> None:
        """Create required database tables."""
        self.execute(
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

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL
            )
            """
        )