"""
Project APEX — Market Selector

Continuously scores all available Deriv Volatility Index symbols across
multiple dimensions (trend clarity, volatility profile, regime) and ranks
them so the Autopilot can dynamically subscribe to only the best markets.

Scoring dimensions:
  1. Trend Clarity (ADX):        High ADX → strategies work better
  2. Volatility Profile (ATR%):  Target a sweet-spot — not too quiet, not chaotic
  3. Regime Bonus:               STRONG_TREND gets a bonus; HIGH_VOLATILITY is penalized
  4. Cooldown Penalty:           Recent stop-loss on a symbol temporarily reduces its score
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from project_apex.indicators.trend import ADX
from project_apex.indicators.volatility import ATR
from project_apex.ai.regime import RegimeDetector, MarketRegime


# All Deriv Volatility Indices available for trading
DERIV_UNIVERSE: list[str] = [
    "R_10", "R_25", "R_50", "R_75", "R_100",
    "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
]

# Sweet-spot ATR% range for optimal trading conditions
# Too low: no movement. Too high: chaotic / unpredictable.
_ATR_PCT_LOW = 0.003   # 0.3% of price
_ATR_PCT_HIGH = 0.025  # 2.5% of price


@dataclass
class MarketScore:
    """Score snapshot for a single symbol."""
    symbol: str
    total_score: float          # 0.0 → 100.0 (higher = better opportunity)
    trend_score: float          # ADX contribution
    volatility_score: float     # ATR-% contribution
    regime_bonus: float         # Regime adjustment
    cooldown_penalty: float     # Recent loss penalty
    regime: str                 # Current detected regime label
    adx: float
    atr_pct: float
    scored_at: float = field(default_factory=time.monotonic)

    def __str__(self) -> str:
        return (
            f"{self.symbol:>12} | score={self.total_score:5.1f} "
            f"trend={self.trend_score:5.1f} vol={self.volatility_score:5.1f} "
            f"regime={self.regime:<14} ADX={self.adx:4.1f} ATR%={self.atr_pct:.3%}"
        )


class MarketScorer:
    """
    Computes an opportunity score for a single symbol given its recent candle history.

    Args:
        adx_threshold: ADX above this value is considered a strong trend.
        atr_low_pct: Lower bound of the ATR-% sweet-spot.
        atr_high_pct: Upper bound of the ATR-% sweet-spot.
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        atr_low_pct: float = _ATR_PCT_LOW,
        atr_high_pct: float = _ATR_PCT_HIGH,
    ) -> None:
        self._adx_threshold = adx_threshold
        self._atr_low = atr_low_pct
        self._atr_high = atr_high_pct
        self._adx_ind = ADX(period=14)
        self._atr_ind = ATR(period=14)
        self._regime_detector = RegimeDetector(adx_threshold=adx_threshold)

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        cooldown_penalty: float = 0.0,
    ) -> MarketScore | None:
        """
        Score a symbol given its OHLCV DataFrame (must have ≥50 rows).

        Args:
            symbol: Symbol name.
            df: DataFrame with columns [timestamp, open, high, low, close].
            cooldown_penalty: Extra penalty (0–40) applied for recent stop-losses.

        Returns:
            MarketScore or None if not enough data.
        """
        if len(df) < 50:
            return None

        try:
            df = self._adx_ind.calculate(df.copy())
            df = self._atr_ind.calculate(df)
        except Exception as exc:
            logger.warning(f"[MarketScorer] Indicator error for {symbol}: {exc}")
            return None

        latest = df.iloc[-1]
        adx = float(latest.get(self._adx_ind.name, 0.0) or 0.0)
        atr = float(latest.get(self._atr_ind.name, 0.0) or 0.0)
        close = float(latest["close"])
        atr_pct = atr / close if close > 0 else 0.0

        # ── Trend clarity score (0–40) ────────────────────────────────────────
        # ADX: 0→25 = noise, 25→50 = trend, 50+ = strong trend (cap at 50)
        trend_score = min(adx, 50.0) / 50.0 * 40.0

        # ── Volatility profile score (0–40) ──────────────────────────────────
        # Target: sweet spot between _atr_low and _atr_high
        # Score peaks at midpoint; drops off toward 0 at either extreme
        mid = (self._atr_low + self._atr_high) / 2
        half_range = (self._atr_high - self._atr_low) / 2
        if half_range > 0:
            dist_from_mid = abs(atr_pct - mid)
            vol_score = max(0.0, 1.0 - (dist_from_mid / half_range)) * 40.0
        else:
            vol_score = 0.0

        # ── Regime bonus/penalty (–20 → +20) ─────────────────────────────────
        regime = self._regime_detector.detect(
            df,
            adx_col=self._adx_ind.name,
            plus_di_col=self._adx_ind.name_plus_di,
            minus_di_col=self._adx_ind.name_minus_di,
            atr_col=self._atr_ind.name,
        )
        regime_bonus = {
            MarketRegime.STRONG_TREND_UP: 20.0,
            MarketRegime.STRONG_TREND_DOWN: 20.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.HIGH_VOLATILITY: -20.0,
        }[regime]

        total = max(0.0, trend_score + vol_score + regime_bonus - cooldown_penalty)

        return MarketScore(
            symbol=symbol,
            total_score=round(total, 2),
            trend_score=round(trend_score, 2),
            volatility_score=round(vol_score, 2),
            regime_bonus=regime_bonus,
            cooldown_penalty=cooldown_penalty,
            regime=regime.value,
            adx=round(adx, 2),
            atr_pct=round(atr_pct, 6),
        )


class MarketSelector:
    """
    Maintains live opportunity scores for all Deriv symbols and exposes
    a ranked list so the Autopilot can pick the best markets to trade.

    Usage::
        selector = MarketSelector(candle_fetcher=my_fetch_fn)
        await selector.update_all()
        top = selector.get_top_symbols(n=2)

    Args:
        candle_fetcher: Async callable ``(symbol, timeframe) → pd.DataFrame``.
            Must be injected from the Application (uses the repository).
        universe: List of symbols to score. Defaults to all Deriv volatility indices.
        timeframe: Candle timeframe (seconds) used for scoring.
        cooldown_s: Seconds after a stop-loss before the cooldown penalty fades.
        cooldown_penalty: Score penalty (0–40) applied during cooldown.
    """

    def __init__(
        self,
        candle_fetcher: Callable,
        universe: list[str] | None = None,
        timeframe: int = 300,
        cooldown_s: float = 1800.0,
        cooldown_penalty: float = 30.0,
    ) -> None:
        self._fetch = candle_fetcher
        self._universe = universe or DERIV_UNIVERSE
        self._timeframe = timeframe
        self._cooldown_s = cooldown_s
        self._cooldown_penalty = cooldown_penalty
        self._scorer = MarketScorer()

        self._scores: dict[str, MarketScore] = {}
        self._last_stop_time: dict[str, float] = {}  # symbol → monotonic time

        logger.info(
            f"[MarketSelector] Initialized | universe={self._universe} | "
            f"timeframe={timeframe}s | cooldown={cooldown_s}s"
        )

    def record_stop_loss(self, symbol: str) -> None:
        """Called by the broker/portfolio when a stop-loss fires on a symbol."""
        self._last_stop_time[symbol] = time.monotonic()
        logger.info(f"[MarketSelector] Cooldown started for {symbol}")

    def _get_cooldown_penalty(self, symbol: str) -> float:
        last = self._last_stop_time.get(symbol, 0.0)
        elapsed = time.monotonic() - last
        if elapsed >= self._cooldown_s:
            return 0.0
        # Linear fade: full penalty at t=0, zero at t=cooldown_s
        return self._cooldown_penalty * (1.0 - elapsed / self._cooldown_s)

    async def update_all(self) -> list[MarketScore]:
        """
        Re-scores all symbols in the universe. Call this every N minutes.
        Returns the updated ranked list.
        """
        scores: list[MarketScore] = []

        for symbol in self._universe:
            try:
                df = await self._fetch(symbol, self._timeframe)
                if df is None or len(df) < 50:
                    continue
                penalty = self._get_cooldown_penalty(symbol)
                score = self._scorer.score(symbol, df, cooldown_penalty=penalty)
                if score is not None:
                    self._scores[symbol] = score
                    scores.append(score)
            except Exception as exc:
                logger.warning(f"[MarketSelector] Failed to score {symbol}: {exc}")

        scores.sort(key=lambda s: s.total_score, reverse=True)

        logger.info(f"[MarketSelector] Ranking updated ({len(scores)} symbols scored):")
        for rank, s in enumerate(scores[:5], 1):
            logger.info(f"  #{rank} {s}")

        return scores

    def get_top_symbols(self, n: int = 2) -> list[str]:
        """Returns the top-N symbol names by current score."""
        ranked = sorted(self._scores.values(), key=lambda s: s.total_score, reverse=True)
        return [s.symbol for s in ranked[:n]]

    def get_all_scores(self) -> list[dict]:
        """Returns all scores as serializable dicts for the dashboard."""
        ranked = sorted(self._scores.values(), key=lambda s: s.total_score, reverse=True)
        return [
            {
                "symbol": s.symbol,
                "total_score": s.total_score,
                "trend_score": s.trend_score,
                "volatility_score": s.volatility_score,
                "regime_bonus": s.regime_bonus,
                "cooldown_penalty": s.cooldown_penalty,
                "regime": s.regime,
                "adx": s.adx,
                "atr_pct": s.atr_pct,
            }
            for s in ranked
        ]
