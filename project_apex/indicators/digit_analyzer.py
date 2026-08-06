"""
Project APEX
Digit Analyzer (Enhanced)

Advanced multi-timeframe digit distribution analyzer with:
- EMA frequency tracking
- Markov chain transition probabilities
- Pattern clustering detection
- Volatility-weighted scoring
- Multi-window analysis (fast/medium/slow)
"""

import math
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

class DigitAnalyzer:
    def __init__(
        self, 
        alpha: float = 0.05,
        fast_window: int = 50,
        medium_window: int = 200,
        slow_window: int = 500
    ):
        """
        Args:
            alpha: EMA smoothing factor (lower = more smoothing)
            fast_window: Recent tick window for short-term patterns
            medium_window: Medium-term pattern window
            slow_window: Long-term baseline window
        """
        self.alpha = alpha
        self.fast_window = fast_window
        self.medium_window = medium_window
        self.slow_window = slow_window
        
        # Track EMA frequency for each digit 0-9 (initialized uniformly at 0.1)
        self.freq: Dict[int, float] = {d: 0.1 for d in range(10)}
        
        # Track ticks since each digit last appeared
        self.drought: Dict[int, int] = {d: 0 for d in range(10)}
        
        # Multi-timeframe windows
        self.recent_digits: deque[int] = deque(maxlen=slow_window)
        
        # Markov chain: transition matrix P(next_digit | current_digit)
        # transitions[i][j] = count of i followed by j
        self.transitions: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in range(10)} for i in range(10)
        }
        
        # Cluster tracking: consecutive repeated digits
        self.current_digit: Optional[int] = None
        self.current_streak: int = 0
        self.max_streak_seen: Dict[int, int] = {d: 0 for d in range(10)}
        
        # Win/loss tracking per excluded digit (for adaptive confidence)
        self.exclusion_wins: Dict[int, int] = {d: 0 for d in range(10)}
        self.exclusion_losses: Dict[int, int] = {d: 0 for d in range(10)}
        
        self.total_ticks = 0
        self.max_decimals = 0
        self.previous_digit: Optional[int] = None
        
        # Volatility proxy: standard deviation of digit changes
        self.digit_changes: deque[float] = deque(maxlen=100)

    def update(self, price: float) -> None:
        """Update frequencies, droughts, transitions, and patterns based on last digit."""
        # Find current decimal places to reconstruct stripped trailing zeros
        price_str = str(price)
        if '.' in price_str:
            decimals = len(price_str.split('.')[1])
        else:
            decimals = 0
            
        if decimals > self.max_decimals:
            self.max_decimals = decimals
            
        # Format the price with the max decimals seen so we don't lose trailing zeros
        if self.max_decimals > 0:
            formatted_price = f"{price:.{self.max_decimals}f}"
        else:
            formatted_price = str(price)
            
        last_digit_char = formatted_price[-1]
        d = int(last_digit_char)
        
        self.total_ticks += 1
        
        # 1. Update Markov transition matrix
        if self.previous_digit is not None:
            self.transitions[self.previous_digit][d] += 1
            
            # Volatility proxy: track how different the new digit is from previous
            digit_change = abs(d - self.previous_digit)
            self.digit_changes.append(float(digit_change))
        
        # 2. Update streak tracking
        if d == self.current_digit:
            self.current_streak += 1
            self.max_streak_seen[d] = max(self.max_streak_seen[d], self.current_streak)
        else:
            self.current_digit = d
            self.current_streak = 1
        
        # 3. Update multi-timeframe windows
        self.recent_digits.append(d)
        
        # 4. Update drought and EMA frequency
        for i in range(10):
            if i == d:
                self.drought[i] = 0
            else:
                self.drought[i] += 1
                
            # Update EMA frequency
            is_match = 1.0 if i == d else 0.0
            self.freq[i] = self.alpha * is_match + (1 - self.alpha) * self.freq[i]
        
        self.previous_digit = d

    def _get_markov_probs(self) -> Dict[int, float]:
        """
        Get the probability that each digit will appear NEXT,
        based on Markov transitions from the current digit.
        """
        if self.current_digit is None:
            return {d: 0.1 for d in range(10)}  # Uniform prior
        
        transitions_from_current = self.transitions[self.current_digit]
        total_transitions = sum(transitions_from_current.values())
        
        if total_transitions == 0:
            return {d: 0.1 for d in range(10)}
        
        return {
            d: count / total_transitions 
            for d, count in transitions_from_current.items()
        }
    
    def _get_window_frequencies(self, window_size: int) -> Dict[int, float]:
        """Get frequency distribution over a specific window."""
        if len(self.recent_digits) < window_size:
            window_size = len(self.recent_digits)
        
        if window_size == 0:
            return {d: 0.1 for d in range(10)}
        
        # Get last N digits
        window = list(self.recent_digits)[-window_size:]
        counts = {d: 0 for d in range(10)}
        
        for digit in window:
            counts[digit] += 1
        
        return {d: count / window_size for d, count in counts.items()}
    
    def _get_volatility_factor(self) -> float:
        """
        Higher volatility (frequent digit changes) = lower confidence.
        Returns a factor between 0.5 (high volatility) and 1.0 (stable).
        """
        if len(self.digit_changes) < 10:
            return 1.0
        
        std_dev = np.std(list(self.digit_changes))
        # Normalize: std_dev ranges from 0 (no change) to ~4.5 (max change)
        # Map to [1.0, 0.5]
        volatility_penalty = max(0.5, 1.0 - (std_dev / 9.0))
        return volatility_penalty
    
    def _get_historical_accuracy(self, digit: int) -> float:
        """
        Returns the win rate for excluding this digit historically.
        If we've never tried this digit, return 0.5 (neutral).
        """
        wins = self.exclusion_wins.get(digit, 0)
        losses = self.exclusion_losses.get(digit, 0)
        total = wins + losses
        
        if total == 0:
            return 0.5  # No history
        
        return wins / total

    def _compute_scores(self) -> Dict[int, float]:
        """
        Compute advanced multi-factor score for each digit.
        Lower score = better candidate for exclusion.
        
        Factors:
        1. EMA frequency (30%) - lower is better
        2. Drought (20%) - longer drought is better
        3. Markov probability (25%) - lower next-tick probability is better
        4. Multi-timeframe consistency (15%) - consistent low appearance across windows
        5. Historical accuracy (10%) - past win rate for excluding this digit
        """
        scores = {}
        
        # Get all factor inputs
        markov_probs = self._get_markov_probs()
        fast_freq = self._get_window_frequencies(self.fast_window)
        medium_freq = self._get_window_frequencies(self.medium_window)
        slow_freq = self._get_window_frequencies(self.slow_window)
        volatility_factor = self._get_volatility_factor()
        
        for d in range(10):
            # Factor 1: EMA frequency (30%)
            ema_score = self.freq[d] * 0.30
            
            # Factor 2: Drought (20%)
            # Inverse: longer drought = lower score
            drought_score = (1.0 / (self.drought[d] + 1)) * 0.20
            
            # Factor 3: Markov next-tick probability (25%)
            markov_score = markov_probs[d] * 0.25
            
            # Factor 4: Multi-timeframe consistency (15%)
            # Average frequency across all windows
            mtf_avg = (fast_freq[d] + medium_freq[d] + slow_freq[d]) / 3.0
            mtf_score = mtf_avg * 0.15
            
            # Factor 5: Historical accuracy penalty (10%)
            # If this digit historically fails when excluded, increase score (make it worse)
            historical_win_rate = self._get_historical_accuracy(d)
            # Invert: lower win rate = higher penalty
            history_penalty = (1.0 - historical_win_rate) * 0.10
            
            # Combine all factors
            raw_score = ema_score + drought_score + markov_score + mtf_score + history_penalty
            
            # Apply volatility factor: in high volatility, all scores become less reliable
            scores[d] = raw_score * volatility_factor
        
        return scores

    def safest_exclusion_digit(self) -> Tuple[int, float]:
        """
        Returns the (digit, confidence_score) that is the safest to exclude in a Differs trade.
        
        Confidence is based on:
        1. Separation from runner-up (how clear the winner is)
        2. Volatility regime (stable markets = higher confidence)
        3. Sufficient warm-up data
        """
        min_warmup = max(50, self.fast_window)
        if self.total_ticks < min_warmup:
            return 0, 0.0
            
        scores = self._compute_scores()
        
        # Sort digits by score (lowest first = best to exclude)
        sorted_digits = sorted(scores.items(), key=lambda x: x[1])
        
        best_digit, best_score = sorted_digits[0]
        runner_up_digit, runner_up_score = sorted_digits[1]
        
        # Base confidence: separation between winner and runner-up
        if runner_up_score > 0 and runner_up_score > best_score:
            separation_confidence = (runner_up_score - best_score) / runner_up_score
        else:
            separation_confidence = 0.0
        
        # Volatility adjustment: reduce confidence in volatile conditions
        volatility_factor = self._get_volatility_factor()
        
        # Historical accuracy boost: if this digit has won often, boost confidence
        historical_accuracy = self._get_historical_accuracy(best_digit)
        accuracy_boost = historical_accuracy * 0.2  # Up to +20% confidence
        
        # Final confidence
        confidence = min(0.99, separation_confidence * volatility_factor + accuracy_boost)
            
        return best_digit, confidence
    
    def record_outcome(self, excluded_digit: int, won: bool) -> None:
        """
        Record the outcome of a Differs trade for this excluded digit.
        Used for adaptive learning.
        """
        if won:
            self.exclusion_wins[excluded_digit] += 1
        else:
            self.exclusion_losses[excluded_digit] += 1

    def hot_digits(self, top_n: int = 3) -> List[int]:
        """Return the N most frequently appearing digits recently."""
        sorted_digits = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in sorted_digits[:top_n]]

    def cold_digits(self, top_n: int = 3) -> List[int]:
        """Return the N least frequently appearing digits recently."""
        sorted_digits = sorted(self.freq.items(), key=lambda x: x[1])
        return [d for d, _ in sorted_digits[:top_n]]
    
    def get_diagnostic_info(self) -> Dict:
        """Return detailed diagnostic information for monitoring."""
        return {
            "total_ticks": self.total_ticks,
            "current_digit": self.current_digit,
            "current_streak": self.current_streak,
            "volatility_factor": self._get_volatility_factor(),
            "ema_frequencies": dict(self.freq),
            "droughts": dict(self.drought),
            "fast_window_freq": self._get_window_frequencies(self.fast_window),
            "medium_window_freq": self._get_window_frequencies(self.medium_window),
            "slow_window_freq": self._get_window_frequencies(self.slow_window),
            "markov_probs": self._get_markov_probs(),
            "historical_accuracy": {
                d: self._get_historical_accuracy(d) for d in range(10)
            }
        }

