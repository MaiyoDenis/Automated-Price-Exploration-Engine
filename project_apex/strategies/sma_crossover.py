import pandas as pd
import numpy as np

from project_apex.strategies.base import Strategy
from project_apex.indicators.moving_averages import SMA


class SMACrossoverStrategy(Strategy):
    """
    A simple Moving Average Crossover strategy.
    Generates a buy signal when the fast SMA crosses above the slow SMA.
    Generates a sell signal when the fast SMA crosses below the slow SMA.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50) -> None:
        super().__init__(f"SMA_Crossover_{fast_period}_{slow_period}")
        
        self.fast_sma = SMA(fast_period)
        self.slow_sma = SMA(slow_period)
        
        self.add_indicator(self.fast_sma)
        self.add_indicator(self.slow_sma)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_data(data)
        
        # Initialize signal column
        df['signal'] = 0
        
        fast_col = self.fast_sma.name
        slow_col = self.slow_sma.name
        
        # We need previous values to detect a crossover
        df['prev_fast'] = df[fast_col].shift(1)
        df['prev_slow'] = df[slow_col].shift(1)
        
        # Buy signal: fast crosses above slow
        buy_condition = (df[fast_col] > df[slow_col]) & (df['prev_fast'] <= df['prev_slow'])
        
        # Sell signal: fast crosses below slow
        sell_condition = (df[fast_col] < df[slow_col]) & (df['prev_fast'] >= df['prev_slow'])
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Drop temporary columns
        df = df.drop(columns=['prev_fast', 'prev_slow'])
        
        return df
