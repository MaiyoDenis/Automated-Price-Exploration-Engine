"""
Project APEX — Full Test Suite for Backtesting Performance Metrics

Covers every function in project_apex/backtesting/metrics.py with both
happy-path and boundary/edge cases.
"""
from __future__ import annotations

import math

import pytest

from project_apex.backtesting.metrics import (
    compute_calmar,
    compute_expectancy,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trades(*profits: float) -> list[dict]:
    return [{"profit": p} for p in profits]


# ===========================================================================
# compute_sharpe
# ===========================================================================

class TestComputeSharpe:
    def test_positive_returns_gives_positive_sharpe(self) -> None:
        returns = [0.01, 0.02, 0.015, 0.03, 0.01]
        result = compute_sharpe(returns)
        assert result > 0

    def test_flat_returns_zero(self) -> None:
        assert compute_sharpe([0.0, 0.0, 0.0]) == 0.0

    def test_single_value_returns_zero(self) -> None:
        assert compute_sharpe([0.05]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_sharpe([]) == 0.0

    def test_zero_std_returns_zero(self) -> None:
        """Identical non-zero returns → std=0 → 0.0 guard."""
        assert compute_sharpe([0.01, 0.01, 0.01]) == 0.0

    def test_known_value(self) -> None:
        """Manual computation: mean=0.01, std=0, single return → 0.0."""
        returns = [0.02, -0.01, 0.03]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance)
        expected = (mean_r / std_r) * math.sqrt(252)
        result = compute_sharpe(returns, periods_per_year=252)
        assert abs(result - expected) < 1e-10

    def test_custom_periods_per_year(self) -> None:
        returns = [0.01, 0.02, -0.005, 0.015]
        daily = compute_sharpe(returns, periods_per_year=252)
        hourly = compute_sharpe(returns, periods_per_year=8760)
        # More periods per year → larger annualisation factor → larger Sharpe
        assert hourly > daily

    def test_negative_mean_gives_negative_sharpe(self) -> None:
        returns = [-0.02, -0.01, -0.03, -0.005]
        result = compute_sharpe(returns)
        assert result < 0


# ===========================================================================
# compute_sortino
# ===========================================================================

class TestComputeSortino:
    def test_flat_returns_zero(self) -> None:
        assert compute_sortino([0.0, 0.0, 0.0]) == 0.0

    def test_single_value_returns_zero(self) -> None:
        assert compute_sortino([0.05]) == 0.0

    def test_no_losses_returns_zero(self) -> None:
        """All-positive returns → downside dev = 0 → 0.0 guard."""
        result = compute_sortino([0.01, 0.02, 0.03])
        assert result == 0.0

    def test_positive_returns_positive_sortino(self) -> None:
        returns = [0.02, -0.005, 0.03, -0.001, 0.025]
        result = compute_sortino(returns)
        assert result > 0

    def test_negative_mean_gives_negative_sortino(self) -> None:
        returns = [-0.02, -0.03, -0.01, 0.001]
        result = compute_sortino(returns)
        assert result < 0

    def test_sortino_gte_sharpe_for_asymmetric_returns(self) -> None:
        """When upside is large and downside is small, Sortino >= Sharpe."""
        # Many large gains, few small losses
        returns = [0.05, 0.04, 0.06, -0.001, 0.03, 0.05, -0.001]
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        # Sortino ignores upside volatility so should be >= sharpe
        assert sortino >= sharpe


# ===========================================================================
# compute_max_drawdown
# ===========================================================================

class TestComputeMaxDrawdown:
    def test_known_drawdown(self) -> None:
        """Peak=110, trough=99 → DD=(110-99)/110 ≈ 0.1."""
        equity = [100.0, 110.0, 99.0, 105.0]
        result = compute_max_drawdown(equity)
        assert abs(result - (110 - 99) / 110) < 1e-10

    def test_monotonic_increase_zero_drawdown(self) -> None:
        equity = [100.0, 110.0, 120.0, 130.0]
        assert compute_max_drawdown(equity) == 0.0

    def test_single_element_zero(self) -> None:
        assert compute_max_drawdown([100.0]) == 0.0

    def test_empty_list_zero(self) -> None:
        assert compute_max_drawdown([]) == 0.0

    def test_full_loss(self) -> None:
        """Equity goes from 100 to 0 → max drawdown = 1.0."""
        equity = [100.0, 50.0, 0.0]
        # (100 - 0) / 100 = 1.0
        assert compute_max_drawdown(equity) == 1.0

    def test_multiple_peaks(self) -> None:
        """Chooses the largest drawdown across multiple peaks."""
        equity = [100.0, 120.0, 80.0, 130.0, 60.0]
        # Drawdown 1: 120→80 = 33.3%
        # Drawdown 2: 130→60 = 53.8%
        result = compute_max_drawdown(equity)
        expected = (130.0 - 60.0) / 130.0
        assert abs(result - expected) < 1e-10

    def test_result_in_0_to_1(self) -> None:
        import random
        random.seed(1)
        equity = [100.0]
        for _ in range(100):
            equity.append(equity[-1] * (1 + random.uniform(-0.05, 0.05)))
        dd = compute_max_drawdown(equity)
        assert 0.0 <= dd <= 1.0


# ===========================================================================
# compute_calmar
# ===========================================================================

class TestComputeCalmar:
    def test_correct_value(self) -> None:
        """10% annual return / 5% max drawdown = calmar=2.0."""
        result = compute_calmar(annual_return_pct=10.0, max_drawdown=0.05)
        assert abs(result - 2.0) < 1e-10

    def test_zero_drawdown_returns_zero(self) -> None:
        assert compute_calmar(10.0, 0.0) == 0.0

    def test_negative_drawdown_returns_zero(self) -> None:
        assert compute_calmar(10.0, -0.1) == 0.0

    def test_negative_return_gives_negative_calmar(self) -> None:
        result = compute_calmar(-10.0, 0.1)
        assert result < 0

    def test_high_calmar_better_than_low(self) -> None:
        calmar_high = compute_calmar(20.0, 0.05)  # 4.0
        calmar_low = compute_calmar(5.0, 0.10)    # 0.5
        assert calmar_high > calmar_low


# ===========================================================================
# compute_win_rate
# ===========================================================================

class TestComputeWinRate:
    def test_half_wins(self) -> None:
        trades = _trades(10.0, -5.0, 20.0, -3.0)
        assert compute_win_rate(trades) == 0.5

    def test_all_wins(self) -> None:
        trades = _trades(10.0, 5.0, 20.0)
        assert compute_win_rate(trades) == 1.0

    def test_all_losses(self) -> None:
        trades = _trades(-10.0, -5.0, -20.0)
        assert compute_win_rate(trades) == 0.0

    def test_zero_profit_not_counted_as_win(self) -> None:
        """Trades with profit=0 are not wins (strictly > 0)."""
        trades = _trades(0.0, 0.0, 10.0)
        # 1 win / 3 trades
        assert abs(compute_win_rate(trades) - 1 / 3) < 1e-10

    def test_empty_list_returns_zero(self) -> None:
        assert compute_win_rate([]) == 0.0

    def test_single_winning_trade(self) -> None:
        assert compute_win_rate(_trades(5.0)) == 1.0

    def test_single_losing_trade(self) -> None:
        assert compute_win_rate(_trades(-5.0)) == 0.0


# ===========================================================================
# compute_profit_factor
# ===========================================================================

class TestComputeProfitFactor:
    def test_correct_value(self) -> None:
        """Gross profit=30, gross loss=5 → PF=6.0."""
        trades = _trades(10.0, -5.0, 20.0)
        assert compute_profit_factor(trades) == 6.0

    def test_no_losses_returns_inf(self) -> None:
        trades = _trades(10.0, 5.0, 20.0)
        assert math.isinf(compute_profit_factor(trades))

    def test_no_profits_returns_zero(self) -> None:
        trades = _trades(-10.0, -5.0)
        assert compute_profit_factor(trades) == 0.0

    def test_empty_list_returns_zero(self) -> None:
        assert compute_profit_factor([]) == 0.0

    def test_balanced_returns_1(self) -> None:
        trades = _trades(10.0, -10.0)
        assert compute_profit_factor(trades) == 1.0


# ===========================================================================
# compute_expectancy
# ===========================================================================

class TestComputeExpectancy:
    def test_correct_value(self) -> None:
        trades = _trades(10.0, -5.0, 20.0, -5.0)
        # (10 - 5 + 20 - 5) / 4 = 5.0
        assert compute_expectancy(trades) == 5.0

    def test_empty_list_returns_zero(self) -> None:
        assert compute_expectancy([]) == 0.0

    def test_single_trade(self) -> None:
        assert compute_expectancy(_trades(7.5)) == 7.5

    def test_negative_expectancy(self) -> None:
        trades = _trades(-10.0, -5.0, 2.0)
        result = compute_expectancy(trades)
        assert result < 0

    def test_zero_expectancy_balanced(self) -> None:
        trades = _trades(5.0, -5.0)
        assert compute_expectancy(trades) == 0.0
