"""
Script to run the Grid Optimizer for Project APEX strategies.

Usage:
    python scripts/optimize.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.config.config import Config
from project_apex.backtesting.optimizer import GridOptimizer

# We'll use a mocked Strategy for this script since LiveStrategies 
# currently differ in initialization from old Backtest Strategies,
# but the concept remains the same.
from project_apex.strategies.base import Strategy
from project_apex.indicators.oscillators import RSI
import pandas as pd
import numpy as np

class BacktestRSIStrategy(Strategy):
    """A simple version of RSI strategy strictly for fast backtesting."""
    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self.period = config.get("period", 14)
        self.oversold = config.get("oversold", 30)
        self.overbought = config.get("overbought", 70)
        self.add_indicator(RSI(period=self.period))
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_data(df)
        df['signal'] = 0
        
        rsi_col = f"RSI_{self.period}"
        if rsi_col not in df.columns:
            return df
            
        # Buy when RSI crosses below oversold, Sell when crosses above overbought
        # simplified for vectorization:
        df['signal'] = np.where(df[rsi_col] < self.oversold, 1, 
                       np.where(df[rsi_col] > self.overbought, -1, 0))
        return df

def main():
    config = Config()
    db_path = config.get_str("database", "path")
    symbol = "R_25"
    timeframe = 60
    
    db = SQLiteManager(db_path)
    db.connect()
    
    optimizer = GridOptimizer(db=db)
    
    param_grid = {
        "period": [10, 14, 21],
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80]
    }
    
    best_params, best_result = optimizer.optimize(
        strategy_class=BacktestRSIStrategy,
        symbol=symbol,
        timeframe=timeframe,
        param_grid=param_grid,
        target_metric="sharpe_ratio"
    )
    
    db.close()


if __name__ == "__main__":
    main()
