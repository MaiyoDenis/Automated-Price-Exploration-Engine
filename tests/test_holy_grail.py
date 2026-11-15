import time
from project_apex.models.tick import Tick
from project_apex.strategies.holy_grail_strategy import (
    HolyGrailStrategy,
    HOLY_GRAIL_PATTERNS,
    ORIGINAL_PATTERNS,
    STRICT_FILTERED_PATTERNS,
    VPS_VALIDATED_PATTERNS,
)


def test_holy_grail_patterns_structure():
    assert len(ORIGINAL_PATTERNS) == 5
    assert len(STRICT_FILTERED_PATTERNS) == 2
    assert len(VPS_VALIDATED_PATTERNS) == 12
    # Combined total without duplicates
    assert len(HOLY_GRAIL_PATTERNS) == 17

    # Original patterns exist
    assert (6, 1, 1) in HOLY_GRAIL_PATTERNS
    assert HOLY_GRAIL_PATTERNS[(6, 1, 1)] == 6
    assert (6, 8, 1) in HOLY_GRAIL_PATTERNS
    assert HOLY_GRAIL_PATTERNS[(6, 8, 1)] == 1
    assert (2, 9, 4) in HOLY_GRAIL_PATTERNS
    assert HOLY_GRAIL_PATTERNS[(2, 9, 4)] == 2
    assert (7, 2, 3) in HOLY_GRAIL_PATTERNS
    assert HOLY_GRAIL_PATTERNS[(7, 2, 3)] == 7
    assert (9, 3, 2) in HOLY_GRAIL_PATTERNS
    assert HOLY_GRAIL_PATTERNS[(9, 3, 2)] == 2


def test_holy_grail_strategy_virtual_loss_activation_and_reset():
    """
    Test the Virtual Loss confirmation filter:
    1. Pattern appears (9 -> 3 -> 2), skipped tick (7).
    2. Tick 5 is NOT 2 -> Virtual win -> Remains INACTIVE.
    3. Pattern appears again (9 -> 3 -> 2), skipped tick (4).
    4. Tick 5 IS 2 -> Virtual loss! -> Marked ACTIVE.
    5. Pattern appears again (9 -> 3 -> 2), skipped tick (8).
    6. Since active, real trade executes (Differs 2), and resets immediately to INACTIVE.
    """
    strategy = HolyGrailStrategy(config={"require_virtual_loss": True})
    now = int(time.time())

    # --- Occurrence 1: (9, 3, 2) followed by dummy 7, then Tick 5 = 8 (Virtual Win) ---
    strategy.on_tick(Tick(symbol="R_100", price=100.9, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.3, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.2, timestamp=now))
    sig1 = strategy.on_tick(Tick(symbol="R_100", price=100.7, timestamp=now))
    assert sig1 is None  # No real trade placed
    assert len(strategy.pending_virtual_checks["R_100"]) == 1

    # Tick 5 arrives with digit 8 (!= 2)
    strategy.on_tick(Tick(symbol="R_100", price=100.8, timestamp=now))
    assert (9, 3, 2) not in strategy.active_patterns["R_100"]  # Still inactive!

    # --- Occurrence 2: (9, 3, 2) followed by dummy 4, then Tick 5 = 2 (Virtual Loss) ---
    strategy.on_tick(Tick(symbol="R_100", price=100.9, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.3, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.2, timestamp=now))
    sig2 = strategy.on_tick(Tick(symbol="R_100", price=100.4, timestamp=now))
    assert sig2 is None  # No real trade placed
    assert len(strategy.pending_virtual_checks["R_100"]) == 1

    # Tick 5 arrives with digit 2 (matches barrier 2 -> Differs contract would lose!)
    strategy.on_tick(Tick(symbol="R_100", price=100.2, timestamp=now))
    assert (9, 3, 2) in strategy.active_patterns["R_100"]  # NOW ACTIVE!

    # --- Occurrence 3: (9, 3, 2) appears now that it is ACTIVE ---
    strategy.on_tick(Tick(symbol="R_100", price=100.9, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.3, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.2, timestamp=now))
    sig3 = strategy.on_tick(Tick(symbol="R_100", price=100.8, timestamp=now))
    
    # Real trade MUST be placed!
    assert sig3 is not None
    assert sig3.barrier == 2
    assert sig3.metadata.get("virtual_loss_confirmed") is True

    # And pattern MUST immediately reset to INACTIVE!
    assert (9, 3, 2) not in strategy.active_patterns["R_100"]


def test_holy_grail_strategy_direct_mode_without_virtual_loss():
    """When require_virtual_loss is False, trades execute immediately."""
    strategy = HolyGrailStrategy(config={"require_virtual_loss": False})
    now = int(time.time())

    strategy.on_tick(Tick(symbol="R_100", price=100.6, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.1, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.1, timestamp=now))
    signal = strategy.on_tick(Tick(symbol="R_100", price=100.0, timestamp=now))

    assert signal is not None
    assert signal.barrier == 6


def test_holy_grail_strategy_strict_filtered_mode():
    strategy = HolyGrailStrategy(config={"strict_mode": True, "require_virtual_loss": False})
    assert len(strategy.patterns) == 2
    assert strategy.strict_mode is True
    assert (9, 3, 2) in strategy.patterns
    assert (7, 2, 3) in strategy.patterns
    assert (6, 1, 1) not in strategy.patterns

    now = int(time.time())
    # (6, 1, 1) should not trigger in strict mode
    strategy.on_tick(Tick(symbol="R_100", price=100.6, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.1, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.1, timestamp=now))
    signal = strategy.on_tick(Tick(symbol="R_100", price=100.0, timestamp=now))
    assert signal is None

    # (9, 3, 2) -> 2 should trigger
    strategy.on_tick(Tick(symbol="R_100", price=100.9, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.3, timestamp=now))
    strategy.on_tick(Tick(symbol="R_100", price=100.2, timestamp=now))
    signal = strategy.on_tick(Tick(symbol="R_100", price=100.7, timestamp=now))
    assert signal is not None
    assert signal.barrier == 2


def test_holy_grail_state_persistence_resume_under_ttl(tmp_path):
    from project_apex.risk.differs_risk_engine import DiffersRiskEngine
    from project_apex.strategies.holy_grail_state import HolyGrailStateManager

    state_file = str(tmp_path / "holy_grail_state.json")
    manager = HolyGrailStateManager(state_file=state_file, ttl_seconds=30.0)

    # Simulate running bot with 1 loss on pattern (9, 3, 2)
    risk_engine = DiffersRiskEngine(use_pattern_recovery=True)
    risk_engine.consecutive_losses = 1
    risk_engine.pattern_consecutive_losses[(9, 3, 2)] = 1
    risk_engine.last_lost_digit = 2

    strategy = HolyGrailStrategy()
    strategy.active_patterns["R_100"] = {(9, 3, 2), (7, 2, 3)}

    # Save state
    manager.save(strategy.get_state(), risk_engine.get_state())

    # Simulate restart 5 seconds later
    restored_data = manager.load()
    assert restored_data is not None

    new_risk = DiffersRiskEngine(use_pattern_recovery=True)
    new_risk.restore_state(restored_data["risk"])
    assert new_risk.consecutive_losses == 1
    assert new_risk.pattern_consecutive_losses[(9, 3, 2)] == 1
    assert new_risk.last_lost_digit == 2

    new_strategy = HolyGrailStrategy()
    new_strategy.restore_state(restored_data["strategy"])
    assert (9, 3, 2) in new_strategy.active_patterns["R_100"]
    assert (7, 2, 3) in new_strategy.active_patterns["R_100"]


def test_holy_grail_state_persistence_expired_over_ttl(tmp_path):
    import json
    from project_apex.strategies.holy_grail_state import HolyGrailStateManager

    state_file = str(tmp_path / "holy_grail_state.json")
    manager = HolyGrailStateManager(state_file=state_file, ttl_seconds=30.0)

    # Write a state with timestamp 35 seconds ago
    old_time = time.time() - 35.0
    payload = {
        "timestamp": old_time,
        "strategy": {"active_patterns": {"R_100": [[9, 3, 2]]}},
        "risk": {"consecutive_losses": 1, "pattern_consecutive_losses": {"(9, 3, 2)": 1}},
    }
    with open(state_file, "w") as f:
        json.dump(payload, f)

    # Loading should detect expiration, return None, and clear file
    loaded = manager.load()
    assert loaded is None
    assert not (tmp_path / "holy_grail_state.json").exists()

