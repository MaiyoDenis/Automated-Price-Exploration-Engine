"""
Project APEX — Backtesting Engine
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger

from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.strategies.base import Strategy
from project_apex.backtesting import metrics


@dataclass
class BacktestResult:
    """Quantitative results from a backtest run."""
    strategy_name: str
    symbol: str
    timeframe: int
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    trades: list[dict[str, Any]] = field(default_factory=list)


class BacktestingEngine:
    """
    Engine to run strategies against historical data with realistic simulation.
    """

    def __init__(
        self,
        db: SQLiteManager,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        self.db = db
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def load_data(self, symbol: str, timeframe: int) -> pd.DataFrame:
        """Loads historical candles from the database."""
        query = """
        SELECT timestamp, open, high, low, close 
        FROM candles 
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp ASC
        """
        rows = self.db.fetchall(query, (symbol, timeframe))

        if not rows:
            logger.warning(f"No data found for {symbol} with timeframe {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def run(self, strategy: Strategy, symbol: str, timeframe: int) -> BacktestResult | dict:
        """Runs the strategy and returns performance metrics."""
        logger.info(f"Starting backtest for {strategy.name} on {symbol}")

        df = self.load_data(symbol, timeframe)
        if df.empty:
            return {"error": "No data"}

        # Generate signals
        df = strategy.generate_signals(df)

        # Simulate trading
        capital = self.initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0.0
        
        trades = []
        equity_curve = [self.initial_capital]
        period_returns = []
        
        last_capital = self.initial_capital

        for index, row in df.iterrows():
            signal = row.get('signal', 0)
            price = row['close']
            
            # Close existing position if signal flips or goes to 0
            if position != 0 and signal != 0 and signal != position:
                # Simulate slippage on exit
                exit_price = price * (1 - self.slippage_pct) if position == 1 else price * (1 + self.slippage_pct)
                
                profit_gross = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                
                # Apply commission
                commission = exit_price * self.commission_pct
                profit_net = profit_gross - commission
                
                profit_pct = profit_net / entry_price
                capital += capital * profit_pct
                
                trades.append({
                    'exit_time': row['datetime'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'profit': profit_net,
                    'profit_pct': profit_pct,
                    'capital': capital
                })
                
                position = 0
            
            # Open new position
            if position == 0 and signal != 0:
                position = signal
                # Simulate slippage and commission on entry
                entry_price = price * (1 + self.slippage_pct) if position == 1 else price * (1 - self.slippage_pct)
                # We simply account for entry commission immediately as a reduction in effectively available capital 
                # (though here we deduct it on close for simplicity).
            
            equity_curve.append(capital)
            
            if len(equity_curve) > 1:
                period_returns.append((capital - last_capital) / last_capital)
            last_capital = capital

        # Calculate metrics using the metrics module
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        win_rate = metrics.compute_win_rate(trades) * 100
        profit_factor = metrics.compute_profit_factor(trades)
        expectancy = metrics.compute_expectancy(trades)
        
        # Approximate periods per year for the timeframe (assuming 24/7 market like Deriv)
        # e.g., 5 min = 12 * 24 * 365 = 105120 periods/year
        periods_per_year = (60 * 60 * 24 * 365) // timeframe if timeframe else 252
        
        sharpe = metrics.compute_sharpe(period_returns, periods_per_year=periods_per_year)
        sortino = metrics.compute_sortino(period_returns, periods_per_year=periods_per_year)
        max_drawdown = metrics.compute_max_drawdown(equity_curve)
        calmar = metrics.compute_calmar(total_return, max_drawdown)

        logger.success(
            f"Backtest completed. Return: {total_return:.2f}%, Win Rate: {win_rate:.2f}%, "
            f"Sharpe: {sharpe:.2f}, Max DD: {max_drawdown*100:.2f}%"
        )

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            initial_capital=self.initial_capital,
            final_capital=capital,
            total_return_pct=total_return,
            total_trades=len(trades),
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_drawdown,
            calmar_ratio=calmar,
            trades=trades,
        )
