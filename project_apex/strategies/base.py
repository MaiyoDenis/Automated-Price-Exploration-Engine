from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

from project_apex.indicators.base import Indicator


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.indicators: List[Indicator] = []

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
            data (pd.DataFrame): DataFrame containing market data and calculated indicators.
                                 
        Returns:
            pd.DataFrame: DataFrame with an added 'signal' column.
                          1 for buy, -1 for sell, 0 for hold.
        """
        pass
