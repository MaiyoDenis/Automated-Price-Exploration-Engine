import pytest
from project_apex.risk.differs_risk_engine import DiffersRiskEngine
from project_apex.strategies.differs_signal import DifferSignal
from project_apex.strategies.signals import SignalType
from project_apex.risk.models import TradeOrder, Direction


def test_recovery_bans_last_lost_digit():
    engine = DiffersRiskEngine(
        base_stake=1.0,
        recovery_multiplier=11.0,
        loss_cooldown_ticks=0, # set 0 to isolate digit ban check
        recovery_min_drought=0,
        ban_lost_digit_on_recovery=True,
    )

    # 1. Simulate a loss on digit 1
    loss_order = TradeOrder(
        symbol="R_25",
        direction=Direction.LONG,
        size=1.0,
        entry_price=100.0,
        stop_loss=0.0,
        take_profit=0.0,
        strategy_name="HolyGrail_Skip1",
        signal_confidence=0.99,
        metadata={"pattern": (6, 8, 1), "barrier": 1}
    )
    engine.on_trade_result(won=False, order=loss_order)
    assert engine.consecutive_losses == 1
    assert engine.last_lost_digit == 1

    # 2. Next signal proposes Differs 1 again -> Must be REJECTED
    signal_same_digit = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1000,
        strategy_name="HolyGrail_Skip1",
        barrier=1,
        duration_ticks=1,
        metadata={"drought": 10, "pattern": (6, 8, 1)}
    )
    decision = engine._evaluate_sync(signal_same_digit)
    assert not decision.approved
    assert "HOLD_BANNED_LOST_DIGIT" in decision.reason

    # 3. Next signal proposes Differs 7 (different digit) -> Must be APPROVED as RECOVERY
    signal_diff_digit = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1001,
        strategy_name="HolyGrail_Skip1",
        barrier=7,
        duration_ticks=1,
        metadata={"drought": 10, "pattern": (7, 2, 3)}
    )
    decision2 = engine._evaluate_sync(signal_diff_digit)
    assert decision2.approved
    assert decision2.order.size == 11.0 # 11x recovery stake
    assert "RECOVERY" in decision2.reason


def test_recovery_requires_min_drought():
    engine = DiffersRiskEngine(
        base_stake=1.0,
        recovery_multiplier=11.0,
        loss_cooldown_ticks=0,
        recovery_min_drought=8,
        ban_lost_digit_on_recovery=True,
    )

    loss_order = TradeOrder(
        symbol="R_25",
        direction=Direction.LONG,
        size=1.0,
        entry_price=100.0,
        stop_loss=0.0,
        take_profit=0.0,
        strategy_name="HolyGrail_Skip1",
        signal_confidence=0.99,
        metadata={"barrier": 2}
    )
    engine.on_trade_result(won=False, order=loss_order)

    # Signal on digit 7, but drought is only 3 (< 8 required) -> Must be REJECTED
    signal_low_drought = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1000,
        strategy_name="HolyGrail_Skip1",
        barrier=7,
        duration_ticks=1,
        metadata={"drought": 3}
    )
    decision = engine._evaluate_sync(signal_low_drought)
    assert not decision.approved
    assert "HOLD_RECOVERY_DROUGHT" in decision.reason

    # Signal on digit 7 with drought 9 (>= 8) -> Must be APPROVED
    signal_high_drought = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1001,
        strategy_name="HolyGrail_Skip1",
        barrier=7,
        duration_ticks=1,
        metadata={"drought": 9}
    )
    decision2 = engine._evaluate_sync(signal_high_drought)
    assert decision2.approved
    assert decision2.order.size == 11.0


def test_pattern_isolated_recovery():
    engine = DiffersRiskEngine(
        base_stake=1.0,
        recovery_multiplier=11.0,
        loss_cooldown_ticks=0,
        recovery_min_drought=0,
        ban_lost_digit_on_recovery=False,
        use_pattern_recovery=True,
    )

    # 1. Pattern (5, 3, 9) loses
    loss_order_a = TradeOrder(
        symbol="R_25",
        direction=Direction.LONG,
        size=1.0,
        entry_price=100.0,
        stop_loss=0.0,
        take_profit=0.0,
        strategy_name="HolyGrail_Skip1",
        signal_confidence=0.99,
        metadata={"pattern": (5, 3, 9), "barrier": 9}
    )
    engine.on_trade_result(won=False, order=loss_order_a)
    assert engine.pattern_consecutive_losses[(5, 3, 9)] == 1

    # 2. Pattern (3, 8, 5) arrives next. It did NOT lose, so it must trade NORMAL stake ($1.00)
    sig_b = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1000,
        strategy_name="HolyGrail_Skip1",
        barrier=6,
        duration_ticks=1,
        metadata={"pattern": (3, 8, 5), "drought": 5}
    )
    dec_b = engine._evaluate_sync(sig_b)
    assert dec_b.approved
    assert dec_b.order.size == 1.0  # Normal trade!
    assert "NORMAL" in dec_b.order.metadata["trade_type"]

    # 3. Pattern (3, 8, 5) wins
    engine.on_trade_result(won=True, order=dec_b.order)
    # Pattern (5, 3, 9) should still have 1 loss
    assert engine.pattern_consecutive_losses[(5, 3, 9)] == 1

    # 4. Pattern (5, 3, 9) arrives again! Now it MUST trade RECOVERY stake ($11.00)
    sig_a_recovery = DifferSignal(
        symbol="R_25",
        signal_type=SignalType.BUY,
        confidence=0.99,
        price=100.0,
        timestamp=1002,
        strategy_name="HolyGrail_Skip1",
        barrier=9,
        duration_ticks=1,
        metadata={"pattern": (5, 3, 9), "drought": 5}
    )
    dec_a = engine._evaluate_sync(sig_a_recovery)
    assert dec_a.approved
    assert dec_a.order.size == 11.0  # 11x Recovery trade!
    assert "RECOVERY" in dec_a.order.metadata["trade_type"]

    # 5. Pattern (5, 3, 9) wins recovery trade
    engine.on_trade_result(won=True, order=dec_a.order)
    assert engine.pattern_consecutive_losses[(5, 3, 9)] == 0

    # 6. Next time (5, 3, 9) arrives, it's back to NORMAL ($1.00)
    dec_a_normal = engine._evaluate_sync(sig_a_recovery)
    assert dec_a_normal.approved
    assert dec_a_normal.order.size == 1.0

