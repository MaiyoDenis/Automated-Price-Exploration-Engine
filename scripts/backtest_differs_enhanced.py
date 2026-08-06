"""
Project APEX - Enhanced Differs Strategy Backtesting

Compares the original vs enhanced Differs strategy performance.
"""
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from project_apex.indicators.digit_analyzer import DigitAnalyzer
from project_apex.database.sqlite_manager import SQLiteManager


class DiffersBacktest:
    """
    Backtest framework for Differs strategy variants.
    """
    
    def __init__(self, db_path: str = "datasets/database.db"):
        self.db = SQLiteManager(db_path)
        
    def load_tick_data(
        self, 
        symbol: str = "R_50", 
        days: int = 7
    ) -> pd.DataFrame:
        """Load tick data from database."""
        import time
        lookback_s = days * 86400
        start_ts = int(time.time()) - lookback_s
        
        query = """
        SELECT timestamp, symbol, price
        FROM ticks
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """
        
        rows = self.db.fetchall(query, (symbol, start_ts))
        if not rows:
            logger.warning(f"No tick data found for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=["timestamp", "symbol", "price"])
        logger.info(f"Loaded {len(df):,} ticks for {symbol}")
        return df
    
    def extract_last_digit(self, price: float, max_decimals: int = 5) -> int:
        """Extract last digit from price."""
        formatted = f"{price:.{max_decimals}f}"
        return int(formatted[-1])
    
    def simulate_differs_trade(
        self,
        excluded_digit: int,
        actual_digits: List[int],
        duration: int = 1
    ) -> bool:
        """
        Simulate a DIGITDIFF trade outcome.
        Returns True if won (excluded digit didn't appear in next N ticks).
        """
        if len(actual_digits) < duration:
            return False  # Not enough data
        
        next_digits = actual_digits[:duration]
        return excluded_digit not in next_digits
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.70,
        duration_ticks: int = 1,
        fast_window: int = 50,
        medium_window: int = 200,
        slow_window: int = 500,
        alpha: float = 0.05,
        min_warmup: int = 100
    ) -> Dict[str, Any]:
        """
        Run enhanced Differs strategy backtest.
        
        Returns performance metrics.
        """
        analyzer = DigitAnalyzer(
            alpha=alpha,
            fast_window=fast_window,
            medium_window=medium_window,
            slow_window=slow_window
        )
        
        trades = []
        signals_generated = 0
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            price = row['price']
            timestamp = row['timestamp']
            
            # Update analyzer
            analyzer.update(price)
            
            # Skip warmup period
            if idx < min_warmup:
                continue
            
            # Get signal
            excluded_digit, confidence = analyzer.safest_exclusion_digit()
            
            # Check if we should trade
            if confidence >= confidence_threshold:
                signals_generated += 1
                
                # Get next N digits for trade simulation
                if idx + duration_ticks < len(df):
                    future_prices = df.iloc[idx+1:idx+1+duration_ticks]['price'].tolist()
                    future_digits = [self.extract_last_digit(p) for p in future_prices]
                    
                    # Simulate trade
                    won = self.simulate_differs_trade(
                        excluded_digit, 
                        future_digits, 
                        duration_ticks
                    )
                    
                    # Record outcome
                    analyzer.record_outcome(excluded_digit, won)
                    
                    trades.append({
                        'timestamp': timestamp,
                        'excluded_digit': excluded_digit,
                        'confidence': confidence,
                        'won': won,
                        'actual_digits': future_digits,
                        'volatility': analyzer._get_volatility_factor(),
                    })
        
        # Calculate metrics
        if not trades:
            return {
                'total_trades': 0,
                'signals_generated': signals_generated,
                'win_rate': 0.0,
                'avg_confidence': 0.0,
            }
        
        trades_df = pd.DataFrame(trades)
        wins = trades_df['won'].sum()
        total = len(trades_df)
        win_rate = wins / total if total > 0 else 0.0
        
        # Per-digit analysis
        per_digit_stats = []
        for digit in range(10):
            digit_trades = trades_df[trades_df['excluded_digit'] == digit]
            if len(digit_trades) > 0:
                per_digit_stats.append({
                    'digit': digit,
                    'trades': len(digit_trades),
                    'wins': digit_trades['won'].sum(),
                    'win_rate': digit_trades['won'].mean(),
                    'avg_confidence': digit_trades['confidence'].mean(),
                })
        
        # Confidence vs win rate analysis
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        trades_df['conf_bin'] = pd.cut(trades_df['confidence'], bins=confidence_bins)
        conf_analysis = trades_df.groupby('conf_bin')['won'].agg(['count', 'sum', 'mean']).to_dict('index')
        
        return {
            'total_trades': total,
            'signals_generated': signals_generated,
            'signal_filter_rate': (signals_generated - total) / signals_generated if signals_generated > 0 else 0.0,
            'wins': wins,
            'losses': total - wins,
            'win_rate': win_rate,
            'avg_confidence': float(trades_df['confidence'].mean()),
            'median_confidence': float(trades_df['confidence'].median()),
            'avg_volatility': float(trades_df['volatility'].mean()),
            'per_digit_stats': per_digit_stats,
            'confidence_analysis': conf_analysis,
            'trades_df': trades_df,
        }
    
    def compare_strategies(self, df: pd.DataFrame) -> None:
        """Compare original vs enhanced strategy."""
        logger.info("\n" + "="*70)
        logger.info("BASELINE STRATEGY (Original - Simple EMA)")
        logger.info("="*70)
        
        # Baseline: higher threshold, no multi-timeframe
        baseline_results = self.run_backtest(
            df,
            confidence_threshold=0.75,
            fast_window=100,
            medium_window=100,
            slow_window=100,
            alpha=0.05,
        )
        
        self._print_results("BASELINE", baseline_results)
        
        logger.info("\n" + "="*70)
        logger.info("ENHANCED STRATEGY (Multi-factor + Adaptive)")
        logger.info("="*70)
        
        # Enhanced: lower threshold, multi-timeframe
        enhanced_results = self.run_backtest(
            df,
            confidence_threshold=0.70,
            fast_window=50,
            medium_window=200,
            slow_window=500,
            alpha=0.05,
        )
        
        self._print_results("ENHANCED", enhanced_results)
        
        # Comparison
        logger.info("\n" + "="*70)
        logger.info("IMPROVEMENT ANALYSIS")
        logger.info("="*70)
        
        if baseline_results['total_trades'] > 0 and enhanced_results['total_trades'] > 0:
            win_rate_improvement = (
                (enhanced_results['win_rate'] - baseline_results['win_rate']) 
                / baseline_results['win_rate'] * 100
            )
            trade_volume_change = (
                (enhanced_results['total_trades'] - baseline_results['total_trades'])
                / baseline_results['total_trades'] * 100
            )
            
            logger.info(f"Win Rate Change: {win_rate_improvement:+.2f}%")
            logger.info(f"Trade Volume Change: {trade_volume_change:+.2f}%")
            logger.info(f"Baseline Expected Value: {baseline_results['win_rate']:.3f}")
            logger.info(f"Enhanced Expected Value: {enhanced_results['win_rate']:.3f}")
            
            # Edge improvement
            baseline_edge = (baseline_results['win_rate'] - 0.5) * 2  # Convert to [-1, 1]
            enhanced_edge = (enhanced_results['win_rate'] - 0.5) * 2
            logger.info(f"Baseline Edge: {baseline_edge:.3f}")
            logger.info(f"Enhanced Edge: {enhanced_edge:.3f}")
            logger.info(f"Edge Improvement: {(enhanced_edge - baseline_edge):.3f}")
    
    def _print_results(self, name: str, results: Dict[str, Any]) -> None:
        """Pretty print backtest results."""
        logger.info(f"\n{name} Results:")
        logger.info(f"  Total Trades: {results['total_trades']:,}")
        logger.info(f"  Signals Generated: {results['signals_generated']:,}")
        logger.info(f"  Win Rate: {results['win_rate']:.2%}")
        logger.info(f"  Wins: {results['wins']:,} | Losses: {results['losses']:,}")
        logger.info(f"  Avg Confidence: {results['avg_confidence']:.3f}")
        logger.info(f"  Median Confidence: {results['median_confidence']:.3f}")
        logger.info(f"  Avg Volatility Factor: {results['avg_volatility']:.3f}")
        
        # Per-digit analysis
        logger.info(f"\n  Per-Digit Performance:")
        for stat in results['per_digit_stats']:
            logger.info(
                f"    Digit {stat['digit']}: "
                f"{stat['trades']:3d} trades, "
                f"win_rate={stat['win_rate']:.2%}, "
                f"conf={stat['avg_confidence']:.3f}"
            )


async def main():
    """Run backtest comparison."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    backtester = DiffersBacktest()
    
    # Load tick data
    logger.info("Loading tick data...")
    df = backtester.load_tick_data(symbol="R_50", days=7)
    
    if df.empty:
        logger.error("No data available. Cannot run backtest.")
        return
    
    # Run comparison
    backtester.compare_strategies(df)


if __name__ == "__main__":
    asyncio.run(main())
