"""Tests for PerformanceTracker."""
from __future__ import annotations

from project_apex.strategies.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    def test_initial_state(self):
        tracker = PerformanceTracker("TestStrat")
        assert tracker.trade_count == 0
        assert tracker.win_rate == 0.5  # Default
        assert tracker.expectancy == 0.0
        assert tracker.max_losing_streak == 0
        assert tracker.weight == 1.0
        assert not tracker.is_disabled

    def test_record_win_boosts_metrics(self):
        tracker = PerformanceTracker("TestStrat")
        tracker.record(pnl=10.0, pnl_pct=0.01)
        assert tracker.trade_count == 1
        assert tracker.win_rate == 1.0
        assert tracker.expectancy == 0.01
        assert tracker.max_losing_streak == 0

    def test_record_loss_increases_losing_streak(self):
        tracker = PerformanceTracker("TestStrat")
        tracker.record(pnl=-10.0, pnl_pct=-0.01)
        tracker.record(pnl=-5.0, pnl_pct=-0.005)
        assert tracker.trade_count == 2
        assert tracker.win_rate == 0.0
        assert tracker.max_losing_streak == 2

    def test_weight_logic(self):
        tracker = PerformanceTracker("TestStrat")
        
        # Not enough data
        for _ in range(5):
            tracker.record(pnl=10.0, pnl_pct=0.01)
        assert tracker.weight == 1.0

        # Good performance -> high weight
        for _ in range(10):
            tracker.record(pnl=10.0, pnl_pct=0.01)
        assert tracker.weight == 1.5

        # Bad performance -> low weight
        for _ in range(20):
            tracker.record(pnl=-10.0, pnl_pct=-0.01)
        assert tracker.weight == 0.25

    def test_disable_logic(self):
        tracker = PerformanceTracker("TestStrat", disable_threshold=0.0, disable_after=3)
        
        # 3 consecutive bad trades
        tracker.record(pnl=-10.0, pnl_pct=-0.01)
        tracker.record(pnl=-10.0, pnl_pct=-0.01)
        tracker.record(pnl=-10.0, pnl_pct=-0.01)
        
        assert tracker.is_disabled
        assert tracker.weight == 0.0

        # One good trade re-enables
        tracker.record(pnl=50.0, pnl_pct=0.05)
        assert not tracker.is_disabled
