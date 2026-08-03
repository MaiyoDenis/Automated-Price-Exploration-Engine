"""
Project APEX — Backtesting Performance Metrics

Pure functions that compute standard quantitative finance performance metrics.
All functions operate on plain Python lists / pandas Series with no side effects.
"""
from __future__ import annotations

import math
from typing import Sequence


def compute_sharpe(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe Ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year).

    Args:
        returns: Sequence of period returns (e.g. daily return fractions).
        risk_free_rate: Annualised risk-free rate (default 0).
        periods_per_year: Trading periods per year (252 for daily, 12 for monthly).

    Returns:
        Sharpe ratio, or 0.0 if not computable.
    """
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    return (mean_r - period_rf) / std_r * math.sqrt(periods_per_year)


def compute_sortino(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino Ratio — like Sharpe but uses only downside deviation.

    Returns:
        Sortino ratio, or 0.0 if not computable.
    """
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    period_rf = risk_free_rate / periods_per_year
    downside = [min(r - period_rf, 0) ** 2 for r in returns]
    downside_dev = math.sqrt(sum(downside) / len(downside)) if downside else 0.0
    if downside_dev == 0:
        return 0.0
    return (mean_r - period_rf) / downside_dev * math.sqrt(periods_per_year)


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    """
    Maximum drawdown = largest peak-to-trough decline in the equity curve.

    Returns:
        Max drawdown as a positive fraction [0, 1].
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_calmar(annual_return_pct: float, max_drawdown: float) -> float:
    """
    Calmar Ratio = annual_return / max_drawdown.

    High Calmar (> 1.0) means the strategy earns more than it risks.

    Returns:
        Calmar ratio, or 0.0 if max_drawdown is 0.
    """
    if max_drawdown <= 0:
        return 0.0
    return annual_return_pct / (max_drawdown * 100)


def compute_win_rate(trades: list[dict]) -> float:
    """
    Win rate = fraction of trades with positive P&L.

    Args:
        trades: List of trade dicts with a 'profit' key.

    Returns:
        Win rate in [0, 1].
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("profit", 0) > 0)
    return wins / len(trades)


def compute_profit_factor(trades: list[dict]) -> float:
    """
    Profit factor = gross_profit / gross_loss.

    > 1.0 = strategy makes more than it loses overall.

    Args:
        trades: List of trade dicts with a 'profit' key.

    Returns:
        Profit factor, or 0.0 if no losses.
    """
    gross_profit = sum(t["profit"] for t in trades if t.get("profit", 0) > 0)
    gross_loss = abs(sum(t["profit"] for t in trades if t.get("profit", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_expectancy(trades: list[dict]) -> float:
    """
    Mathematical expectancy = mean profit per trade.

    Positive expectancy is the minimum requirement for a viable strategy.
    """
    if not trades:
        return 0.0
    return sum(t.get("profit", 0) for t in trades) / len(trades)
