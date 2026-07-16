import pandas as pd
from project_apex.indicators.base import Indicator


class SMA(Indicator):
    """
    Simple Moving Average (SMA).
    """

    def __init__(self, period: int, column: str = "close") -> None:
        self.period = period
        self.column = column
        self.name = f"sma_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")
            
        data[self.name] = data[self.column].rolling(window=self.period).mean()
        return data


class EMA(Indicator):
    """
    Exponential Moving Average (EMA).
    """

    def __init__(self, period: int, column: str = "close") -> None:
        self.period = period
        self.column = column
        self.name = f"ema_{period}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data.")
            
        data[self.name] = data[self.column].ewm(span=self.period, adjust=False).mean()
        return data
