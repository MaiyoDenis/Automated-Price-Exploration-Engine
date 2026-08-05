"""Tests for MarketScorer and MarketSelector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project_apex.intelligence.market_selector import MarketScorer, MarketSelector, DERIV_UNIVERSE


def _make_trending_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic trending candle DataFrame."""
    price = 1000.0
    rows = []
    for i in range(n):
        price += np.random.normal(0.5, 0.3)  # Upward drift
        rows.append({
            "timestamp": i * 300,
            "open": price - 0.1,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
        })
    return pd.DataFrame(rows)


def _make_ranging_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic ranging candle DataFrame."""
    price = 1000.0
    rows = []
    for i in range(n):
        price += np.random.normal(0, 0.3)  # No drift
        price = max(995.0, min(1005.0, price))  # Bounded
        rows.append({
            "timestamp": i * 300,
            "open": price - 0.1,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
        })
    return pd.DataFrame(rows)


class TestMarketScorer:
    def test_score_returns_market_score_for_valid_data(self):
        scorer = MarketScorer()
        df = _make_trending_df(100)
        score = scorer.score("R_50", df)
        assert score is not None
        assert score.symbol == "R_50"
        assert 0 <= score.total_score <= 100

    def test_score_returns_none_for_insufficient_data(self):
        scorer = MarketScorer()
        df = _make_trending_df(30)  # Less than minimum 50
        score = scorer.score("R_50", df)
        assert score is None

    def test_cooldown_penalty_reduces_score(self):
        scorer = MarketScorer()
        df = _make_trending_df(100)
        score_no_penalty = scorer.score("R_50", df, cooldown_penalty=0.0)
        score_with_penalty = scorer.score("R_50", df, cooldown_penalty=30.0)
        assert score_no_penalty is not None
        assert score_with_penalty is not None
        assert score_no_penalty.total_score > score_with_penalty.total_score

    def test_score_components_are_non_negative(self):
        scorer = MarketScorer()
        df = _make_trending_df(100)
        score = scorer.score("R_25", df)
        assert score is not None
        assert score.trend_score >= 0
        assert score.volatility_score >= 0

    def test_score_includes_regime_label(self):
        scorer = MarketScorer()
        df = _make_trending_df(100)
        score = scorer.score("R_75", df)
        assert score is not None
        assert score.regime in ("TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE")


class TestMarketSelector:
    @pytest.mark.asyncio
    async def test_update_all_returns_sorted_scores(self):
        """MarketSelector should return scores sorted by total_score descending."""

        async def fake_fetch(symbol: str, timeframe: int) -> pd.DataFrame:
            return _make_trending_df(100)

        selector = MarketSelector(candle_fetcher=fake_fetch, universe=["R_25", "R_50", "R_75"])
        scores = await selector.update_all()
        totals = [s.total_score for s in scores]
        assert totals == sorted(totals, reverse=True)

    @pytest.mark.asyncio
    async def test_get_top_symbols_returns_correct_count(self):
        async def fake_fetch(symbol: str, timeframe: int) -> pd.DataFrame:
            return _make_trending_df(100)

        selector = MarketSelector(candle_fetcher=fake_fetch, universe=["R_25", "R_50", "R_75"])
        await selector.update_all()
        top = selector.get_top_symbols(n=2)
        assert len(top) == 2
        assert all(s in ["R_25", "R_50", "R_75"] for s in top)

    @pytest.mark.asyncio
    async def test_cooldown_applied_after_stop_loss(self):
        async def fake_fetch(symbol: str, timeframe: int) -> pd.DataFrame:
            return _make_trending_df(100)

        selector = MarketSelector(
            candle_fetcher=fake_fetch,
            universe=["R_25", "R_50"],
            cooldown_s=3600.0,
            cooldown_penalty=50.0,
        )
        await selector.update_all()
        score_before = selector._scores.get("R_25")

        selector.record_stop_loss("R_25")
        await selector.update_all()
        score_after = selector._scores.get("R_25")

        assert score_before is not None and score_after is not None
        assert score_after.total_score < score_before.total_score

    @pytest.mark.asyncio
    async def test_handles_fetch_errors_gracefully(self):
        async def failing_fetch(symbol: str, timeframe: int) -> pd.DataFrame:
            if symbol == "R_25":
                raise RuntimeError("Connection error")
            return _make_trending_df(100)

        selector = MarketSelector(
            candle_fetcher=failing_fetch,
            universe=["R_25", "R_50"],
        )
        scores = await selector.update_all()
        # R_25 failed but R_50 should still be scored
        assert len(scores) >= 1
        assert all(s.symbol != "R_25" for s in scores)

    def test_get_all_scores_serializable(self):
        """get_all_scores() must return plain dicts with no custom types."""
        import asyncio

        async def fake_fetch(symbol: str, timeframe: int) -> pd.DataFrame:
            return _make_trending_df(100)

        selector = MarketSelector(candle_fetcher=fake_fetch, universe=["R_50"])
        asyncio.run(selector.update_all())
        scores = selector.get_all_scores()
        assert isinstance(scores, list)
        if scores:
            import json
            json.dumps(scores)  # Should not raise
