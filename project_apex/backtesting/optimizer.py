"""
Project APEX — Strategy Optimizer

Runs combinations of parameters against the BacktestingEngine to find 
the optimal configuration for a given symbol and timeframe.
"""
from __future__ import annotations

import itertools
from typing import Any, Type

from loguru import logger

from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.strategies.base import Strategy
from project_apex.backtesting.engine import BacktestingEngine, BacktestResult


class GridOptimizer:
    """
    Performs an exhaustive search through a specified parameter grid.
    """

    def __init__(self, db: SQLiteManager, initial_capital: float = 10000.0) -> None:
        self.engine = BacktestingEngine(db=db, initial_capital=initial_capital)

    def optimize(
        self,
        strategy_class: Type[Strategy],
        symbol: str,
        timeframe: int,
        param_grid: dict[str, list[Any]],
        target_metric: str = "sharpe_ratio"
    ) -> tuple[dict[str, Any] | None, BacktestResult | None]:
        """
        Runs the strategy for every combination in param_grid.
        
        Args:
            strategy_class: The Strategy class to instantiate (must accept **kwargs or a config dict).
            symbol: Trading instrument.
            timeframe: Candle timeframe.
            param_grid: Dictionary of parameter names to lists of possible values.
            target_metric: The BacktestResult attribute to maximize (e.g. 'sharpe_ratio', 'total_return_pct').
            
        Returns:
            Tuple of (best_parameters, best_result).
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = list(itertools.product(*values))
        total_runs = len(combinations)
        
        logger.info(f"[Optimizer] Starting grid search for {strategy_class.__name__} on {symbol}")
        logger.info(f"[Optimizer] Total combinations to test: {total_runs}")
        
        best_metric = -float('inf')
        best_params = None
        best_result = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            # Instantiate strategy. Assuming __init__ can accept config (or we adapt it)
            # Since standard `Strategy` (for backtesting) is being used, we might need to 
            # ensure it supports dynamic param injection. 
            # For this MVP, we assume `Strategy` takes config dict or kwargs.
            try:
                strategy = strategy_class(name=f"Opt_{i}", config=params)
                result = self.engine.run(strategy, symbol, timeframe)
                
                if isinstance(result, BacktestResult):
                    metric_val = getattr(result, target_metric, 0.0)
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_params = params
                        best_result = result
                        
            except Exception as e:
                logger.error(f"[Optimizer] Failed run {i} with params {params}: {e}")
                
        if best_params:
            logger.success(f"[Optimizer] Best parameters found: {best_params}")
            logger.success(f"[Optimizer] Best {target_metric}: {best_metric:.4f}")
            
        return best_params, best_result
