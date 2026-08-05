"""Tests for CircuitBreaker."""
from __future__ import annotations

import pytest

from project_apex.risk.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert not cb.is_halted
        assert cb.halt_reason == ""

    def test_consecutive_stop_loss_trigger(self):
        cb = CircuitBreaker(halt_on_consecutive_sl=3)
        cb.record_trade_closed(is_stop_loss=True, equity=10000.0)
        cb.record_trade_closed(is_stop_loss=True, equity=9900.0)
        assert not cb.is_halted
        
        cb.record_trade_closed(is_stop_loss=True, equity=9800.0)
        assert cb.is_halted
        assert "CONSECUTIVE_SL" in cb.halt_reason

    def test_win_resets_sl_streak(self):
        cb = CircuitBreaker(halt_on_consecutive_sl=3)
        cb.record_trade_closed(is_stop_loss=True, equity=10000.0)
        cb.record_trade_closed(is_stop_loss=True, equity=9900.0)
        cb.record_trade_closed(is_stop_loss=False, equity=10100.0)  # Win
        cb.record_trade_closed(is_stop_loss=True, equity=10000.0)
        assert not cb.is_halted

    def test_loss_velocity_trigger(self):
        cb = CircuitBreaker(loss_velocity_pct=0.02, velocity_window_s=1800.0)
        cb.update_equity(10000.0)
        cb.update_equity(9900.0)  # -1%
        assert not cb.is_halted

        cb.update_equity(9750.0)  # -2.5% total
        assert cb.is_halted
        assert "LOSS_VELOCITY" in cb.halt_reason

    def test_trade_frequency_trigger(self):
        cb = CircuitBreaker(max_trades_per_hour=3)
        for _ in range(3):
            cb.record_trade_opened()
        assert not cb.is_halted
        
        cb.record_trade_opened()
        assert cb.is_halted
        assert "TRADE_FREQUENCY" in cb.halt_reason

    def test_reset(self):
        cb = CircuitBreaker(halt_on_consecutive_sl=1)
        cb.record_trade_closed(is_stop_loss=True, equity=10000.0)
        assert cb.is_halted
        
        cb.reset()
        assert not cb.is_halted
        assert cb.halt_reason == ""
