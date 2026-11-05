"""
Project APEX
Database Manager — Robust & Self-Healing
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from loguru import logger


class SQLiteManager:
    """SQLite database manager with integrity checks and automatic corruption recovery."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the SQLite database and check integrity."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute("PRAGMA busy_timeout=5000;")
        
        # Verify integrity on connect
        if not self._check_integrity():
            logger.error(f"[SQLiteManager] Database at {self.db_path} is corrupt. Initiating recovery...")
            self._repair_or_reset()

    def _check_integrity(self) -> bool:
        """Runs PRAGMA quick_check to verify database health."""
        if self.connection is None:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA quick_check;")
            row = cursor.fetchone()
            return row is not None and row[0] == "ok"
        except Exception as e:
            logger.warning(f"[SQLiteManager] Integrity check failed: {e}")
            return False

    def _repair_or_reset(self) -> None:
        """Backs up corrupt database file and recreates clean schema."""
        self.close()
        db_file = Path(self.db_path)
        if db_file.exists():
            backup_path = db_file.with_name(f"{db_file.name}.corrupt.{int(time.time())}")
            try:
                db_file.rename(backup_path)
                logger.info(f"[SQLiteManager] Moved corrupt database to {backup_path}")
            except Exception as e:
                logger.error(f"[SQLiteManager] Failed to rename corrupt database: {e}")

        # Remove leftover WAL and SHM files
        for ext in ["-wal", "-shm"]:
            wal_file = Path(f"{self.db_path}{ext}")
            if wal_file.exists():
                try:
                    wal_file.unlink()
                except Exception:
                    pass

        # Re-connect fresh
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.execute("PRAGMA busy_timeout=5000;")
        self.initialize()
        logger.success("[SQLiteManager] Database auto-recovery completed successfully.")

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

    def execute(self, query: str, params: tuple = ()) -> None:
        """Execute INSERT/UPDATE/DELETE queries with automatic recovery on malformed DB."""
        if self.connection is None:
            raise RuntimeError("Database is not connected.")

        try:
            with self.connection:
                self.connection.execute(query, params)
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error(f"[SQLiteManager] Malformed database detected during execute: {e}")
                self._repair_or_reset()
                with self.connection:
                    self.connection.execute(query, params)
            else:
                raise

    def execute_many(self, query: str, params: list[tuple]) -> None:
        """Execute INSERT/UPDATE/DELETE queries with multiple parameters."""
        if self.connection is None:
            raise RuntimeError("Database is not connected.")

        try:
            with self.connection:
                self.connection.executemany(query, params)
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error(f"[SQLiteManager] Malformed database detected during execute_many: {e}")
                self._repair_or_reset()
                with self.connection:
                    self.connection.executemany(query, params)
            else:
                raise

    def fetchall(self, query: str, params: tuple = ()) -> list:
        """Execute SELECT queries."""
        if self.connection is None:
            raise RuntimeError("Database is not connected.")

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.error(f"[SQLiteManager] Malformed database detected during fetchall: {e}")
                self._repair_or_reset()
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
            else:
                raise

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

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                id INTEGER PRIMARY KEY DEFAULT 1,
                is_halted INTEGER NOT NULL,
                halt_until REAL NOT NULL,
                active_trigger TEXT NOT NULL,
                reason TEXT NOT NULL,
                last_updated REAL NOT NULL,
                CONSTRAINT single_row CHECK (id = 1)
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy_name TEXT PRIMARY KEY,
                wins INTEGER NOT NULL,
                total_trades INTEGER NOT NULL,
                expectancy REAL NOT NULL,
                win_rate REAL NOT NULL,
                rolling_sharpe REAL NOT NULL
            )
            """
        )