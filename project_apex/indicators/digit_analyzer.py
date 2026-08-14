"""
Project APEX
Digit Analyzer (Enhanced v3)

Advanced multi-timeframe digit distribution analyzer with:
- EMA frequency tracking (faster alpha=0.08)
- First AND second-order Markov chain transition probabilities (bigrams)
- Streak-aware next-tick penalty
- Pattern clustering detection
- Entropy-based normalised confidence score
- Volatility-weighted scoring
- Multi-window analysis (fast/medium/slow)
- Historical accuracy feedback
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional


class DigitAnalyzer:
    def __init__(
        self,
        alpha: float = 0.08,           # Faster EMA — was 0.05
        fast_window: int = 30,          # Shorter — was 50
        medium_window: int = 150,       # Shorter — was 200
        slow_window: int = 500
    ):
        """
        Args:
            alpha: EMA smoothing factor (higher = faster adaptation to recent digits)
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
        self.recent_digits: deque = deque(maxlen=slow_window)

        # ── Markov chains ──────────────────────────────────────────────────
        # First-order: P(next | current)
        self.transitions: Dict[int, Dict[int, int]] = {
            i: {j: 0 for j in range(10)} for i in range(10)
        }

        # Second-order (bigram): P(next | prev-2, prev-1)
        # bigram_transitions[(a, b)][c] = count of (a,b) → c
        self.bigram_transitions: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(
            lambda: {j: 0 for j in range(10)}
        )

        # ── Streak tracking ────────────────────────────────────────────────
        self.current_digit: Optional[int] = None
        self.current_streak: int = 0
        self.max_streak_seen: Dict[int, int] = {d: 0 for d in range(10)}

        # ── Win/loss tracking per excluded digit ───────────────────────────
        self.exclusion_wins: Dict[int, int] = {d: 0 for d in range(10)}
        self.exclusion_losses: Dict[int, int] = {d: 0 for d in range(10)}

        self.total_ticks = 0
        self.max_decimals = 0
        self.previous_digit: Optional[int] = None
        self.prev_prev_digit: Optional[int] = None   # Two steps back (for bigrams)

        # Volatility proxy: standard deviation of digit-to-digit changes
        self.digit_changes: deque = deque(maxlen=100)

    # ──────────────────────────────────────────────────────────────────────
    # Core update
    # ──────────────────────────────────────────────────────────────────────

    def update(self, price: float) -> None:
        """Update frequencies, droughts, Markov chains, and streak state."""
        price_str = str(price)
        if '.' in price_str:
            decimals = len(price_str.split('.')[1])
        else:
            decimals = 0

        if decimals > self.max_decimals:
            self.max_decimals = decimals

        if self.max_decimals > 0:
            formatted_price = f"{price:.{self.max_decimals}f}"
        else:
            formatted_price = str(price)

        d = int(formatted_price[-1])
        self.total_ticks += 1

        # 1. First-order Markov
        if self.previous_digit is not None:
            self.transitions[self.previous_digit][d] += 1
            self.digit_changes.append(float(abs(d - self.previous_digit)))

        # 2. Second-order Markov (bigram)
        if self.prev_prev_digit is not None and self.previous_digit is not None:
            bigram_key = (self.prev_prev_digit, self.previous_digit)
            self.bigram_transitions[bigram_key][d] += 1

        # 3. Streak tracking
        if d == self.current_digit:
            self.current_streak += 1
            self.max_streak_seen[d] = max(self.max_streak_seen[d], self.current_streak)
        else:
            self.current_digit = d
            self.current_streak = 1

        # 4. Multi-timeframe window
        self.recent_digits.append(d)

        # 5. Drought and EMA frequency
        for i in range(10):
            if i == d:
                self.drought[i] = 0
            else:
                self.drought[i] += 1
            is_match = 1.0 if i == d else 0.0
            self.freq[i] = self.alpha * is_match + (1 - self.alpha) * self.freq[i]

        # 6. Advance history pointers
        self.prev_prev_digit = self.previous_digit
        self.previous_digit = d

    # ──────────────────────────────────────────────────────────────────────
    # Markov probability helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_markov_probs(self) -> Dict[int, float]:
        """First-order Markov: P(next digit | current digit)."""
        if self.current_digit is None:
            return {d: 0.1 for d in range(10)}

        row = self.transitions[self.current_digit]
        total = sum(row.values())
        if total == 0:
            return {d: 0.1 for d in range(10)}

        return {d: count / total for d, count in row.items()}

    def _get_bigram_probs(self) -> Dict[int, float]:
        """
        Second-order Markov: P(next digit | last two digits).
        Falls back to first-order if insufficient bigram history.
        """
        if self.prev_prev_digit is None or self.previous_digit is None:
            return self._get_markov_probs()

        key = (self.prev_prev_digit, self.previous_digit)
        row = self.bigram_transitions.get(key)
        if row is None:
            return self._get_markov_probs()

        total = sum(row.values())
        if total < 5:   # Not enough data for this bigram; fall back
            return self._get_markov_probs()

        return {d: count / total for d, count in row.items()}

    def _get_window_frequencies(self, window_size: int) -> Dict[int, float]:
        """Get frequency distribution over the most recent N ticks."""
        available = len(self.recent_digits)
        if available == 0:
            return {d: 0.1 for d in range(10)}

        actual = min(window_size, available)
        window = list(self.recent_digits)[-actual:]
        counts = {d: 0 for d in range(10)}
        for digit in window:
            counts[digit] += 1
        return {d: count / actual for d, count in counts.items()}

    # ──────────────────────────────────────────────────────────────────────
    # Volatility, accuracy & streak helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_volatility_factor(self) -> float:
        """
        Higher volatility → lower confidence.
        Returns a factor in [0.5, 1.0].
        """
        if len(self.digit_changes) < 10:
            return 1.0
        std_dev = float(np.std(list(self.digit_changes)))
        # Milder penalty: std_dev of ~4.0 drops factor to ~0.85 instead of ~0.55
        return max(0.5, 1.0 - (std_dev / 25.0))

    def _get_historical_accuracy(self, digit: int) -> float:
        """Win rate for excluding this digit. Returns 0.5 if no history."""
        wins = self.exclusion_wins.get(digit, 0)
        losses = self.exclusion_losses.get(digit, 0)
        total = wins + losses
        if total == 0:
            return 0.5
        return wins / total

    def _get_streak_penalty(self, digit: int) -> float:
        """
        If the *current* streak digit is being considered for exclusion,
        add a small penalty because a hot streak means it may repeat.
        Range [0.0, 0.05].
        """
        if digit != self.current_digit:
            return 0.0
        return min(self.current_streak / 60.0, 0.05)

    # ──────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────

    def _compute_scores(self) -> Dict[int, float]:
        """
        Multi-factor score for each digit (lower = better exclusion candidate).

        Weight breakdown (v3):
          EMA frequency       20%  (lower frequency → lower score → better)
          Drought             15%  (longer drought → lower score)
          1st-order Markov    20%  (lower next-tick prob → lower score)
          2nd-order Markov    20%  (richer context, same logic)
          MTF consistency     15%  (consistent low appearance across windows)
          Historical accuracy 10%  (poor historical win rate → higher penalty)
          Streak penalty       +   (additive if digit is on a current streak)
        """
        markov1 = self._get_markov_probs()
        markov2 = self._get_bigram_probs()
        fast_freq = self._get_window_frequencies(self.fast_window)
        medium_freq = self._get_window_frequencies(self.medium_window)
        slow_freq = self._get_window_frequencies(self.slow_window)
        volatility_factor = self._get_volatility_factor()

        scores = {}
        for d in range(10):
            ema_score      = self.freq[d] * 0.20
            
            # Drought: a long drought makes a digit slightly more "due" — gentle signal only
            drought_score  = (1.0 / (self.drought[d] + 1)) * 0.15
                
            markov1_score  = markov1[d] * 0.20
            markov2_score  = markov2[d] * 0.20
            mtf_avg        = (fast_freq[d] + medium_freq[d] + slow_freq[d]) / 3.0
            mtf_score      = mtf_avg * 0.15
            hist_penalty   = (1.0 - self._get_historical_accuracy(d)) * 0.10
            streak_penalty = self._get_streak_penalty(d)

            raw = (
                ema_score + drought_score
                + markov1_score + markov2_score
                + mtf_score + hist_penalty
                + streak_penalty
            )
            scores[d] = raw * volatility_factor

        return scores

    # ──────────────────────────────────────────────────────────────────────
    # Entropy-based confidence
    # ──────────────────────────────────────────────────────────────────────

    def _compute_confidence(self, scores: Dict[int, float], best_digit: int) -> float:
        """
        Hybrid confidence: separation between best and runner-up, boosted by
        a Markov edge term that rewards cases where both 1st- and 2nd-order
        Markov agree the best digit has low next-tick probability.

        Components:
        - Separation (60%): how much the best digit stands out from the pack.
        - Markov edge (40%): combined advantage from 1st and 2nd order Markov.

        Range: [0.0, 1.0]
        """
        sorted_digits = sorted(scores.items(), key=lambda x: x[1])
        best_score = sorted_digits[0][1]
        runner_up_score = sorted_digits[1][1]

        # Separation confidence
        if runner_up_score > 0 and runner_up_score > best_score:
            sep_conf = (runner_up_score - best_score) / runner_up_score
        else:
            sep_conf = 0.0

        # Markov edge: how far below the uniform baseline (0.1) the best digit
        # sits in both Markov orders. Clamp to [0, 1].
        m1 = self._get_markov_probs()
        m2 = self._get_bigram_probs()
        uniform = 1.0 / 10
        # Advantage = max(0, how much lower than random)
        m1_edge = max(0.0, uniform - m1[best_digit]) / uniform   # 0→1
        m2_edge = max(0.0, uniform - m2[best_digit]) / uniform   # 0→1
        markov_edge = (m1_edge + m2_edge) / 2.0

        confidence = sep_conf * 0.60 + markov_edge * 0.40
        return max(0.0, min(1.0, confidence))


    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def safest_exclusion_digit(self, banned_digit: Optional[int] = None) -> Tuple[int, float]:
        """
        Returns (digit, confidence) for the safest DIGITDIFF exclusion.

        Confidence is a hybrid of:
        - Separation (60%): how clearly the best digit stands above the rest.
        - Markov edge (40%): how low the best digit's 1st+2nd-order probability is.
        Dampened by volatility and boosted by historical accuracy.
        Requires at least `min_warmup` ticks (20, reduced from 50 for faster start).
        """
        min_warmup = max(100, self.fast_window)  # Need enough data for Markov to be meaningful
        if self.total_ticks < min_warmup:
            return 0, 0.0

        scores = self._compute_scores()
        
        # PREVENT BACK-TO-BACK: Force the banned digit to have an infinite score
        if banned_digit is not None and banned_digit in scores:
            scores[banned_digit] = float('inf')
            
        sorted_digits = sorted(scores.items(), key=lambda x: x[1])
        best_digit, _ = sorted_digits[0]

        base_conf = self._compute_confidence(scores, best_digit)
        volatility_factor = self._get_volatility_factor()
        accuracy_boost = self._get_historical_accuracy(best_digit) * 0.20  # up to +20%

        confidence = min(0.99, base_conf * volatility_factor + accuracy_boost)
        return best_digit, confidence


    def record_outcome(self, excluded_digit: int, won: bool) -> None:
        """Record the outcome of a Differs trade for adaptive learning."""
        if won:
            self.exclusion_wins[excluded_digit] += 1
        else:
            self.exclusion_losses[excluded_digit] += 1

    def hot_digits(self, top_n: int = 3) -> List[int]:
        """Return the N most frequently appearing digits (EMA-based)."""
        return [d for d, _ in sorted(self.freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def cold_digits(self, top_n: int = 3) -> List[int]:
        """Return the N least frequently appearing digits (EMA-based)."""
        return [d for d, _ in sorted(self.freq.items(), key=lambda x: x[1])[:top_n]]

    def get_diagnostic_info(self) -> Dict:
        """Return detailed diagnostic information for monitoring."""
        scores = (
            self._compute_scores()
            if self.total_ticks >= max(20, self.fast_window)
            else {}
        )
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
            "markov_probs": self._get_markov_probs(),           # backward-compat
            "markov_probs_1st": self._get_markov_probs(),
            "markov_probs_2nd": self._get_bigram_probs(),
            "historical_accuracy": {d: self._get_historical_accuracy(d) for d in range(10)},
            "scores": scores,
        }
