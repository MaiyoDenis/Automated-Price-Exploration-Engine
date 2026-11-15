"""
Project APEX
Holy Grail Skip-1 Strategy

Implements the "Holy Grail" Skip-1 Digit Differs strategy on the live tick stream.
"""
from typing import Any, Dict
from collections import deque
from loguru import logger

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.differs_signal import DifferSignal
from project_apex.strategies.signals import SignalType


# Original 5 Holy Grail Patterns
ORIGINAL_PATTERNS = {
    (6, 1, 1): 6,
    (6, 8, 1): 1,
    (2, 9, 4): 2,
    (7, 2, 3): 7,
    (9, 3, 2): 2,
}

# Strict Filtered Mode Patterns: Retained top historical winners (>= 91% - 96.5% WR)
STRICT_FILTERED_PATTERNS = {
    (9, 3, 2): 2,  # Retained original winner (96.55% WR)
    (7, 2, 3): 7,  # Retained original winner (91.30% WR)
}

# VPS-validated patterns (>=95% - 100% win rate on live tick data)
VPS_VALIDATED_PATTERNS = {
    (5, 3, 9): 9,
    (7, 8, 5): 1,
    (1, 1, 8): 2,
    (3, 8, 5): 6,
    (0, 6, 2): 0,
    (2, 0, 2): 0,
    (9, 4, 1): 4,
    (0, 4, 7): 0,
    (6, 9, 1): 8,
    (3, 1, 3): 1,
    (9, 2, 4): 2,
    (2, 9, 1): 4,
}

# Full combined set of all Holy Grail patterns (17 patterns total)
HOLY_GRAIL_PATTERNS = {
    **ORIGINAL_PATTERNS,
    **VPS_VALIDATED_PATTERNS,
}


class HolyGrailStrategy(LiveStrategy):
    """
    Looks for specific 3-tick patterns, waits for 1 dummy tick, 
    and executes a Differs contract on the 5th tick.

    Includes a Virtual Loss filter:
    Patterns start INACTIVE. When a pattern appears, tick 5 is monitored.
    If tick 5 would have caused a loss (digit matches barrier), the pattern
    becomes ACTIVE. On its next appearance, a real trade is placed and the
    pattern immediately resets to INACTIVE.
    """
    def __init__(self, name: str = "HolyGrail_Skip1", config: Dict[str, Any] = None):
        self.tick_windows: Dict[str, deque] = {}
        self.droughts: Dict[str, Dict[int, int]] = {}
        self.duration_ticks = 1
        self.strict_mode = False
        self.require_virtual_loss = True
        self.patterns = HOLY_GRAIL_PATTERNS
        self.active_patterns: Dict[str, set] = {}
        self.pending_virtual_checks: Dict[str, list] = {}
        self.recent_trades: Dict[str, list] = {}
        self.on_state_change = None
        super().__init__(name, config)
        
    def initialize(self, config: Dict[str, Any]) -> None:
        self.duration_ticks = config.get("duration_ticks", 1)
        self.strict_mode = config.get("strict_mode", False) or config.get("filtered_mode", False)
        self.require_virtual_loss = config.get("require_virtual_loss", True)
        if self.strict_mode:
            self.patterns = STRICT_FILTERED_PATTERNS
            logger.info(f"[{self.name}] Initialized Holy Grail bot in STRICT FILTERED MODE ({len(self.patterns)} patterns).")
        else:
            self.patterns = HOLY_GRAIL_PATTERNS
            logger.info(f"[{self.name}] Initialized Holy Grail bot with all {len(self.patterns)} patterns.")

        if self.require_virtual_loss:
            logger.info(f"[{self.name}] Virtual Loss Filter ENABLED: real trades only execute after virtual loss confirmation.")
        
    def _get_last_digit(self, price: float) -> int:
        return int(str(price)[-1])

    def on_tick(self, tick: Tick) -> DifferSignal | None:
        if tick.symbol not in self.tick_windows:
            self.tick_windows[tick.symbol] = deque(maxlen=4)
            self.droughts[tick.symbol] = {d: 0 for d in range(10)}
            self.recent_trades[tick.symbol] = []
            self.active_patterns[tick.symbol] = set()
            self.pending_virtual_checks[tick.symbol] = []
            
        digit = self._get_last_digit(tick.price)

        # 1. Evaluate any pending virtual trade checks on tick 5
        if self.pending_virtual_checks.get(tick.symbol):
            remaining_checks = []
            for check in self.pending_virtual_checks[tick.symbol]:
                check["ticks_remaining"] -= 1
                if check["ticks_remaining"] <= 0:
                    pat = check["pattern"]
                    tgt = check["target_digit"]
                    if digit == tgt:
                        # Tick 5 matched barrier -> Virtual trade would have LOST!
                        self.active_patterns[tick.symbol].add(pat)
                        logger.info(
                            f"[{self.name}] ⚡ Virtual loss detected for {tick.symbol} on pattern {pat}! "
                            f"(Tick 5 digit {digit} matched barrier {tgt}). Pattern is now ACTIVE for next occurrence."
                        )
                        if callable(self.on_state_change):
                            try:
                                self.on_state_change()
                            except Exception:
                                pass
                    else:
                        logger.debug(
                            f"[{self.name}] Virtual win for {tick.symbol} on pattern {pat} "
                            f"(Tick 5 digit {digit} != barrier {tgt}). Pattern remains inactive."
                        )
                else:
                    remaining_checks.append(check)
            self.pending_virtual_checks[tick.symbol] = remaining_checks

        # 2. Append digit to window & update droughts
        window = self.tick_windows[tick.symbol]
        window.append(digit)
        
        # Update drought tracking
        for d in range(10):
            if d == digit:
                self.droughts[tick.symbol][d] = 0
            else:
                self.droughts[tick.symbol][d] += 1
        
        # 3. Check for pattern match when window has 4 ticks (3 pattern ticks + 1 skipped tick)
        if len(window) == 4:
            t1, t2, t3, t4 = window[0], window[1], window[2], window[3]
            pattern = (t1, t2, t3)
            
            if pattern in self.patterns:
                target_digit = self.patterns[pattern]
                target_drought = self.droughts[tick.symbol].get(target_digit, 0)
                is_active = pattern in self.active_patterns[tick.symbol]

                # If Virtual Loss filter is enabled and pattern is not active yet:
                if self.require_virtual_loss and not is_active:
                    logger.info(
                        f"[{self.name}] ⏳ Pattern Match {t1}->{t2}->{t3} (Ignored: {t4}) for {tick.symbol} [INACTIVE]. "
                        f"Monitoring next tick for virtual loss against barrier {target_digit}..."
                    )
                    self.pending_virtual_checks[tick.symbol].append({
                        "pattern": pattern,
                        "target_digit": target_digit,
                        "ticks_remaining": 1
                    })
                    window.clear()
                    return None

                # Pattern is ACTIVE: Execute real money trade!
                logger.info(
                    f"[{self.name}] 🔥 Pattern Match {t1}->{t2}->{t3} (Ignored: {t4}) for {tick.symbol} [ACTIVE]. "
                    f"Executing Differs {target_digit} (drought={target_drought})"
                )
                
                # Reset pattern immediately back to INACTIVE
                if is_active:
                    self.active_patterns[tick.symbol].remove(pattern)
                    logger.info(f"[{self.name}] 🔄 Pattern {pattern} immediately reset to INACTIVE.")
                    if callable(self.on_state_change):
                        try:
                            self.on_state_change()
                        except Exception:
                            pass
                
                # Clear the window to avoid multiple triggers on overlapping patterns
                window.clear()
                
                return DifferSignal(
                    symbol=tick.symbol,
                    signal_type=SignalType.BUY,
                    confidence=0.99, # Holy grail is high confidence
                    price=tick.price,
                    timestamp=tick.timestamp,
                    strategy_name=self.name,
                    barrier=target_digit,
                    duration_ticks=self.duration_ticks,
                    metadata={
                        "safe_digit": target_digit,
                        "confidence": 0.99,
                        "threshold": 0.90,
                        "banned_digit": None,
                        "pattern": pattern,
                        "drought": target_drought,
                        "virtual_loss_confirmed": True,
                    }
                )
                
        return None
        
    def record_trade_outcome(self, symbol: str, excluded_digit: int, won: bool, confidence: float) -> None:
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        self.recent_trades[symbol].append((excluded_digit, won, confidence))
        
        # Trim to avoid memory leaks
        if len(self.recent_trades[symbol]) > 100:
            self.recent_trades[symbol] = self.recent_trades[symbol][-100:]

    def on_candle(self, candle: Candle) -> None:
        pass
        
    def get_performance_summary(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self.recent_trades or not self.recent_trades[symbol]:
            return {"symbol": symbol, "trades": 0, "win_rate": 0.0}
            
        trades = self.recent_trades[symbol]
        wins = sum(1 for _, won, _ in trades if won)
        return {
            "symbol": symbol,
            "trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": wins / len(trades),
            "avg_confidence": 0.99
        }

    def get_state(self) -> Dict[str, Any]:
        """Export active patterns state for persistence."""
        return {
            "active_patterns": {
                sym: [list(p) for p in patterns]
                for sym, patterns in self.active_patterns.items()
            }
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore active patterns from persistent storage."""
        raw_active = state.get("active_patterns", {})
        total_restored = 0
        for sym, patterns in raw_active.items():
            if sym not in self.active_patterns:
                self.active_patterns[sym] = set()
            for p in patterns:
                self.active_patterns[sym].add(tuple(p))
                total_restored += 1
        logger.info(f"[{self.name}] Restored {total_restored} active patterns across {len(raw_active)} symbols.")

    def reset_state(self) -> None:
        """Reset all active patterns back to inactive."""
        for sym in self.active_patterns:
            self.active_patterns[sym].clear()
        self.pending_virtual_checks.clear()
        logger.info(f"[{self.name}] All active patterns reset to INACTIVE.")

 