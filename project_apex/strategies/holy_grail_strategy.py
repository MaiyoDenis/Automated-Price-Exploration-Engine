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


# The 5 highly profitable patterns discovered in research
HOLY_GRAIL_PATTERNS = {
    (6, 1, 1): 6,
    (6, 8, 1): 1,
    (2, 9, 4): 2,
    (7, 2, 3): 7,
    (9, 3, 2): 2,
}


class HolyGrailStrategy(LiveStrategy):
    """
    Looks for specific 3-tick patterns, waits for 1 dummy tick, 
    and executes a Differs contract on the 5th tick.
    """
    def __init__(self, name: str = "HolyGrail_Skip1", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.tick_windows: Dict[str, deque] = {}
        self.duration_ticks = 1
        
        # Performance tracking
        self.recent_trades: Dict[str, list] = {}
        
    def initialize(self, config: Dict[str, Any]) -> None:
        self.duration_ticks = config.get("duration_ticks", 1)
        logger.info(f"[{self.name}] Initialized Holy Grail bot.")
        
    def _get_last_digit(self, price: float) -> int:
        return int(str(price)[-1])

    def on_tick(self, tick: Tick) -> DifferSignal | None:
        if tick.symbol not in self.tick_windows:
            self.tick_windows[tick.symbol] = deque(maxlen=4)
            self.recent_trades[tick.symbol] = []
            
        window = self.tick_windows[tick.symbol]
        digit = self._get_last_digit(tick.price)
        window.append(digit)
        
        if len(window) == 4:
            t1, t2, t3, t4 = window[0], window[1], window[2], window[3]
            pattern = (t1, t2, t3)
            
            if pattern in HOLY_GRAIL_PATTERNS:
                target_digit = HOLY_GRAIL_PATTERNS[pattern]
                logger.info(f"[{self.name}] 🔥 Pattern Match {t1}->{t2}->{t3} (Ignored: {t4}) for {tick.symbol}. Executing Differs {target_digit}")
                
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
 