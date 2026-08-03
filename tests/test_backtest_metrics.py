"""
Tests for backtesting metrics.
"""
from project_apex.backtesting.metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_win_rate,
    compute_profit_factor
)

def test_sharpe():
    returns = [0.01, 0.02, -0.01, 0.03]
    sharpe = compute_sharpe(returns, periods_per_year=252)
    assert sharpe > 0
    
    returns_flat = [0.0, 0.0, 0.0]
    assert compute_sharpe(returns_flat) == 0.0

def test_max_drawdown():
    equity = [100.0, 110.0, 99.0, 105.0]
    # Peak is 110, trough is 99 -> (110 - 99)/110 = 0.1
    dd = compute_max_drawdown(equity)
    assert dd == 0.1

def test_win_rate():
    trades = [
        {"profit": 10},
        {"profit": -5},
        {"profit": 20},
        {"profit": 0}
    ]
    # 2 wins / 4 trades = 0.5
    wr = compute_win_rate(trades)
    assert wr == 0.5

def test_profit_factor():
    trades = [
        {"profit": 10},
        {"profit": -5},
        {"profit": 20},
        {"profit": 0}
    ]
    # Gross profit = 30, Gross loss = 5
    # PF = 30 / 5 = 6.0
    pf = compute_profit_factor(trades)
    assert pf == 6.0
