import pandas as pd
from loguru import logger
from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.strategies.base import Strategy


class BacktestingEngine:
    """
    Engine to run strategies against historical data.
    """

    def __init__(self, db: SQLiteManager, initial_capital: float = 10000.0) -> None:
        self.db = db
        self.initial_capital = initial_capital
        
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

    def run(self, strategy: Strategy, symbol: str, timeframe: int) -> dict:
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
        
        for index, row in df.iterrows():
            signal = row.get('signal', 0)
            price = row['close']
            
            # Close existing position if signal flips or goes to 0 (depending on logic, let's say we only reverse or close)
            # For simplicity, let's assume signal 1 = go long, -1 = go short, 0 = do nothing.
            
            if position != 0 and signal != 0 and signal != position:
                # Close position
                profit = (price - entry_price) if position == 1 else (entry_price - price)
                profit_pct = profit / entry_price
                capital += capital * profit_pct
                
                trades.append({
                    'exit_time': row['datetime'],
                    'exit_price': price,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'capital': capital
                })
                
                position = 0
            
            # Open new position
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades) * 100 if trades else 0.0
        
        logger.success(f"Backtest completed. Return: {total_return:.2f}%, Win Rate: {win_rate:.2f}%")
        
        return {
            "initial_capital": self.initial_capital,
            "final_capital": capital,
            "total_return_pct": total_return,
            "total_trades": len(trades),
            "win_rate_pct": win_rate,
            "trades": trades
        }
