"""
Project APEX
Differs Strategy (Enhanced)

Advanced tick-level strategy for DIGITDIFF contracts with:
- Multi-factor digit analysis (Markov chains, multi-timeframe, volatility)
- Adaptive confidence thresholds based on market regime
- Performance tracking and feedback learning
- Kelly Criterion position sizing
"""
from typing import Any, Dict, Optional

from loguru import logger

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.differs_signal import DifferSignal
from project_apex.strategies.signals import SignalType
from project_apex.indicators.digit_analyzer import DigitAnalyzer


class DiffersStrategy(LiveStrategy):
    """
    Enhanced tick-level strategy that uses advanced statistical analysis
    to identify the safest digit to exclude in DIGITDIFF contracts.
    """

    def __init__(self, name: str = "Differs_Adaptive_V2", config: Dict[str, Any] = None):
        super().__init__(name, config)
        # We maintain a DigitAnalyzer per symbol
        self.analyzers: Dict[str, DigitAnalyzer] = {}
        
        # Performance tracking for adaptive learning
        self.recent_trades: Dict[str, list] = {}  # symbol -> [(digit, won, confidence), ...]
        self.win_rate_window: int = 50  # Track last N trades for win rate
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize strategy with configuration.
        
        Config keys:
            base_confidence: Minimum confidence threshold (default: 0.70)
            adaptive_confidence: Whether to adjust threshold based on performance (default: True)
            duration_ticks: Contract duration in ticks (default: 1)
            ema_alpha: EMA smoothing factor (default: 0.05)
            fast_window: Short-term window size (default: 50)
            medium_window: Medium-term window size (default: 200)
            slow_window: Long-term window size (default: 500)
            max_confidence_adjustment: Max +/- adjustment to confidence (default: 0.15)
        """
        self.base_confidence = config.get("base_confidence", 0.70)
        self.adaptive_confidence = config.get("adaptive_confidence", True)
        self.duration_ticks = config.get("duration_ticks", 1)
        self.alpha = config.get("ema_alpha", 0.05)
        self.fast_window = config.get("fast_window", 50)
        self.medium_window = config.get("medium_window", 200)
        self.slow_window = config.get("slow_window", 500)
        self.max_conf_adjustment = config.get("max_confidence_adjustment", 0.15)
        
        logger.info(
            f"[{self.name}] Initialized | "
            f"base_conf={self.base_confidence:.2f} "
            f"adaptive={self.adaptive_confidence} "
            f"duration={self.duration_ticks} "
            f"windows=[{self.fast_window},{self.medium_window},{self.slow_window}]"
        )

    def _get_adaptive_threshold(self, symbol: str) -> float:
        """
        Calculate adaptive confidence threshold based on recent performance.
        
        If we're winning more than 60%, lower threshold (be more aggressive).
        If we're winning less than 50%, raise threshold (be more conservative).
        """
        if not self.adaptive_confidence:
            return self.base_confidence
        
        if symbol not in self.recent_trades or len(self.recent_trades[symbol]) < 10:
            return self.base_confidence
        
        # Calculate recent win rate
        recent = self.recent_trades[symbol][-self.win_rate_window:]
        wins = sum(1 for _, won, _ in recent if won)
        win_rate = wins / len(recent)
        
        # Adjustment logic:
        # win_rate > 0.60 → lower threshold (up to -0.15)
        # win_rate < 0.50 → raise threshold (up to +0.15)
        # win_rate = 0.55 → no adjustment
        target_win_rate = 0.55
        adjustment = (target_win_rate - win_rate) * 0.5  # Scale factor
        adjustment = max(-self.max_conf_adjustment, min(self.max_conf_adjustment, adjustment))
        
        adapted_threshold = self.base_confidence + adjustment
        
        # Clamp to reasonable bounds
        adapted_threshold = max(0.50, min(0.95, adapted_threshold))
        
        if len(recent) % 25 == 0:  # Log periodically
            logger.info(
                f"[{self.name}] Adaptive threshold | "
                f"symbol={symbol} win_rate={win_rate:.2%} "
                f"threshold={self.base_confidence:.2f}→{adapted_threshold:.2f}"
            )
        
        return adapted_threshold

    def on_tick(self, tick: Tick) -> DifferSignal | None:
        """
        Process every tick to update the digit analyzer and emit a signal if confident.
        
        Uses enhanced multi-factor analysis with adaptive thresholds.
        """
        # Initialize analyzer for this symbol if needed
        if tick.symbol not in self.analyzers:
            self.analyzers[tick.symbol] = DigitAnalyzer(
                alpha=self.alpha,
                fast_window=self.fast_window,
                medium_window=self.medium_window,
                slow_window=self.slow_window
            )
            self.recent_trades[tick.symbol] = []
        
        analyzer = self.analyzers[tick.symbol]
        analyzer.update(tick.price)
        
        # Get the safest digit to exclude with confidence score
        safe_digit, confidence = analyzer.safest_exclusion_digit()
        
        # Get adaptive threshold based on recent performance
        threshold = self._get_adaptive_threshold(tick.symbol)
        
        # Only trade when confidence exceeds adaptive threshold
        if confidence >= threshold:
            # Get diagnostic info for metadata
            diagnostics = analyzer.get_diagnostic_info()
            
            logger.debug(
                f"[{self.name}] Signal generated | "
                f"symbol={tick.symbol} exclude_digit={safe_digit} "
                f"confidence={confidence:.3f} threshold={threshold:.3f} "
                f"volatility={diagnostics['volatility_factor']:.3f}"
            )
            
            return DifferSignal(
                symbol=tick.symbol,
                signal_type=SignalType.BUY,  # DIGITDIFF is always a "Buy" of a diff contract
                confidence=confidence,
                price=tick.price,
                timestamp=tick.timestamp,
                strategy_name=self.name,
                barrier=safe_digit,
                duration_ticks=self.duration_ticks,
                metadata={
                    "safe_digit": safe_digit,
                    "confidence": confidence,
                    "threshold": threshold,
                    "total_ticks": analyzer.total_ticks,
                    "volatility_factor": diagnostics["volatility_factor"],
                    "current_streak": diagnostics["current_streak"],
                    "current_digit": diagnostics["current_digit"],
                    "ema_freq": diagnostics["ema_frequencies"][safe_digit],
                    "drought": diagnostics["droughts"][safe_digit],
                    "markov_prob": diagnostics["markov_probs"][safe_digit],
                }
            )
        
        return None
    
    def record_trade_outcome(self, symbol: str, excluded_digit: int, won: bool, confidence: float) -> None:
        """
        Record the outcome of a trade for adaptive learning.
        Should be called by the execution engine after contract settlement.
        """
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        
        self.recent_trades[symbol].append((excluded_digit, won, confidence))
        
        # Also update the analyzer's historical tracking
        if symbol in self.analyzers:
            self.analyzers[symbol].record_outcome(excluded_digit, won)
        
        # Trim to window size
        if len(self.recent_trades[symbol]) > self.win_rate_window * 2:
            self.recent_trades[symbol] = self.recent_trades[symbol][-self.win_rate_window:]

    def on_candle(self, candle: Candle) -> None:
        """Differs Strategy operates solely on ticks. Ignore candles."""
        pass
    
    def get_performance_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Return a performance summary for a given symbol.
        Useful for monitoring and diagnostics.
        """
        if symbol not in self.recent_trades or not self.recent_trades[symbol]:
            return {"symbol": symbol, "trades": 0, "win_rate": 0.0}
        
        trades = self.recent_trades[symbol]
        wins = sum(1 for _, won, _ in trades if won)
        avg_confidence = sum(conf for _, _, conf in trades) / len(trades)
        
        # Per-digit win rates
        per_digit: Dict[int, Dict[str, int]] = {d: {"wins": 0, "total": 0} for d in range(10)}
        for digit, won, _ in trades:
            per_digit[digit]["total"] += 1
            if won:
                per_digit[digit]["wins"] += 1
        
        digit_win_rates = {
            d: (stats["wins"] / stats["total"]) if stats["total"] > 0 else None
            for d, stats in per_digit.items()
        }
        
        return {
            "symbol": symbol,
            "trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": wins / len(trades),
            "avg_confidence": avg_confidence,
            "adaptive_threshold": self._get_adaptive_threshold(symbol),
            "per_digit_win_rates": digit_win_rates,
        }
