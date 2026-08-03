"""
Project APEX — Strategy Base Class

All live trading strategies inherit from ``LiveStrategy``.
Backtesting-only strategies may still use the original ``Strategy`` class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from project_apex.indicators.base import Indicator
from project_apex.models.tick import Tick
from project_apex.models.candle import Candle
from project_apex.strategies.signals import TradeSignal


class Strategy(ABC):
    """
    Abstract base class for backtesting strategies (batch signal generation).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.indicators: list[Indicator] = []

    def add_indicator(self, indicator: Indicator) -> None:
        """Adds an indicator to be calculated before signal generation."""
        self.indicators.append(indicator)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates all indicators for the given data."""
        df = data.copy()
        for indicator in self.indicators:
            df = indicator.calculate(df)
        return df

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates buy/sell signals based on the provided data.

        Args:
            data: DataFrame containing market data and calculated indicators.

        Returns:
            DataFrame with an added ``signal`` column.
            1 for buy, -1 for sell, 0 for hold.
        """
        pass


class LiveStrategy(ABC):
    """
    Abstract base class for live-trading strategies.

    Live strategies receive events one at a time (tick or candle) and emit
    a :class:`~project_apex.strategies.signals.TradeSignal` or ``None``.
    They maintain their own indicator state internally.

    Subclasses **must** implement :meth:`on_candle`.
    Implementing :meth:`on_tick` is optional (default: ``None``).
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config: dict[str, Any] = config or {}
        self._candle_history: dict[tuple[str, int], list[Candle]] = {}
        self.initialize(self.config)

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Called once at strategy instantiation.
        Override to set up parameters from config.
        """

    def on_tick(self, tick: Tick) -> TradeSignal | None:
        """
        Called for every validated tick.
        Default implementation returns ``None`` (no signal from raw ticks).
        Override for tick-level strategies.
        """
        return None

    @abstractmethod
    def on_candle(self, candle: Candle) -> TradeSignal | None:
        """
        Called whenever a candle is completed by the CandleBuilder.

        Args:
            candle: The completed candle object.

        Returns:
            A :class:`TradeSignal` if the strategy has a view, else ``None``.
        """

    def _append_candle(self, candle: Candle) -> list[Candle]:
        """
        Internal helper: appends a candle to the per-(symbol, timeframe) history
        and returns the full history list.
        """
        key = (candle.symbol, candle.timeframe)
        if key not in self._candle_history:
            self._candle_history[key] = []
        self._candle_history[key].append(candle)
        return self._candle_history[key]

    def _get_history_df(self, candle: Candle) -> pd.DataFrame:
        """
        Returns the candle history for a given (symbol, timeframe) as a DataFrame.
        The latest candle is already appended via :meth:`_append_candle`.
        """
        history = self._candle_history.get((candle.symbol, candle.timeframe), [])
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "tick_count": c.tick_count,
                }
                for c in history
            ]
        )
