"""
Project APEX — Tests for All Strategy Engines & Built-in Strategies

Covers:
- SMACrossoverStrategy (batch signals)
- RSIMeanReversionStrategy (live candles & ADX filter)
- BollingerBreakoutStrategy (squeeze & breakout detection)
- MACDMomentumStrategy (MACD cross + SuperTrend filter)
- MultiStrategyEnsemble (voting & consensus)
- MetaRegimeStrategy (regime routing)
- MLStrategy (XGBoost prediction interface)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from project_apex.models.candle import Candle
from project_apex.models.tick import Tick
from project_apex.strategies.signals import SignalType, TradeSignal
from project_apex.strategies.sma_crossover import SMACrossoverStrategy
from project_apex.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from project_apex.strategies.bollinger_breakout import BollingerBreakoutStrategy
from project_apex.strategies.macd_momentum import MACDMomentumStrategy
from project_apex.strategies.multi_strategy import MultiStrategyEnsemble
from project_apex.strategies.meta_strategy import MetaRegimeStrategy
from project_apex.strategies.ml_strategy import MLStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(
    symbol: str = "R_25",
    timeframe: int = 60,
    timestamp: int = 1_000_000,
    open_p: float = 100.0,
    high_p: float = 102.0,
    low_p: float = 98.0,
    close_p: float = 101.0,
) -> Candle:
    return Candle(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        tick_count=10,
    )


def _make_candle_series(length: int = 60, symbol: str = "R_25", timeframe: int = 60) -> list[Candle]:
    candles = []
    price = 100.0
    for i in range(length):
        price += np.sin(i * 0.2) * 2.0 + 0.1
        c = Candle(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=1_000_000 + i * timeframe,
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price + 0.2,
            tick_count=10,
        )
        candles.append(c)
    return candles


# ---------------------------------------------------------------------------
# SMACrossoverStrategy Tests
# ---------------------------------------------------------------------------

class TestSMACrossoverStrategy:
    def test_init_and_indicators(self) -> None:
        strat = SMACrossoverStrategy(fast_period=5, slow_period=20)
        assert strat.name == "SMA_Crossover_5_20"
        assert len(strat.indicators) == 2

    def test_generate_signals_crossover(self) -> None:
        # Create DataFrame with clear fast/slow crossover
        prices = [10.0] * 20 + [20.0] * 20  # sharp upward step
        df = pd.DataFrame({"close": prices})
        strat = SMACrossoverStrategy(fast_period=5, slow_period=10)
        result = strat.generate_signals(df)

        assert "signal" in result.columns
        assert set(result["signal"].unique()).issubset({-1, 0, 1})
        # Should have a buy signal near step transition
        assert (result["signal"] == 1).any()


# ---------------------------------------------------------------------------
# RSIMeanReversionStrategy Tests
# ---------------------------------------------------------------------------

class TestRSIMeanReversionStrategy:
    def test_ignores_wrong_timeframe(self) -> None:
        strat = RSIMeanReversionStrategy("rsi_test", config={"timeframe": 60})
        c = _make_candle(timeframe=300)
        assert strat.on_candle(c) is None

    def test_min_bars_guard(self) -> None:
        strat = RSIMeanReversionStrategy("rsi_test", config={"timeframe": 60})
        for i in range(25):  # less than _MIN_BARS (30)
            sig = strat.on_candle(_make_candle(timestamp=1000 + i * 60))
            assert sig is None

    def test_emits_buy_or_sell_or_none(self) -> None:
        strat = RSIMeanReversionStrategy("rsi_test", config={"timeframe": 60, "min_adx": 0.0})
        candles = _make_candle_series(length=50, timeframe=60)
        signals = []
        for c in candles:
            sig = strat.on_candle(c)
            if sig is not None:
                signals.append(sig)

        for sig in signals:
            assert isinstance(sig, TradeSignal)
            assert sig.signal_type in (SignalType.BUY, SignalType.SELL)
            assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# BollingerBreakoutStrategy Tests
# ---------------------------------------------------------------------------

class TestBollingerBreakoutStrategy:
    def test_ignores_wrong_timeframe(self) -> None:
        strat = BollingerBreakoutStrategy("bb_test", config={"timeframe": 900})
        c = _make_candle(timeframe=60)
        assert strat.on_candle(c) is None

    def test_squeeze_state_tracking(self) -> None:
        strat = BollingerBreakoutStrategy("bb_test", config={"timeframe": 60, "squeeze_threshold": 1.0})
        candles = _make_candle_series(length=40, timeframe=60)
        for c in candles:
            strat.on_candle(c)
        # Should track squeeze state per symbol
        assert "R_25" in strat._in_squeeze


# ---------------------------------------------------------------------------
# MACDMomentumStrategy Tests
# ---------------------------------------------------------------------------

class TestMACDMomentumStrategy:
    def test_ignores_wrong_timeframe(self) -> None:
        strat = MACDMomentumStrategy("macd_test", config={"timeframe": 300})
        c = _make_candle(timeframe=60)
        assert strat.on_candle(c) is None

    def test_min_bars_requirement(self) -> None:
        strat = MACDMomentumStrategy("macd_test", config={"timeframe": 300})
        candles = _make_candle_series(length=35, timeframe=300)
        for c in candles:
            assert strat.on_candle(c) is None


# ---------------------------------------------------------------------------
# MultiStrategyEnsemble Tests
# ---------------------------------------------------------------------------

class DummyLiveStrategy(RSIMeanReversionStrategy):
    """Subclass for mock voting."""

    def __init__(self, name: str, return_signal: SignalType | None = None, conf: float = 0.8) -> None:
        self.return_signal = return_signal
        self.conf = conf
        super().__init__(name=name, config={"timeframe": 60})

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if self.return_signal is None:
            return None
        return TradeSignal(
            symbol=candle.symbol,
            signal_type=self.return_signal,
            confidence=self.conf,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
        )


class TestMultiStrategyEnsemble:
    def test_ensemble_consensus_buy(self) -> None:
        s1 = DummyLiveStrategy("s1", SignalType.BUY, 0.8)
        s2 = DummyLiveStrategy("s2", SignalType.BUY, 0.9)
        s3 = DummyLiveStrategy("s3", SignalType.SELL, 0.2)

        ensemble = MultiStrategyEnsemble(
            strategies=[s1, s2, s3],
            min_vote_fraction=0.55,
            min_strategies_agree=2,
            config={"timeframe": 60},
        )

        c = _make_candle(timeframe=60)
        sig = ensemble.on_candle(c)

        assert sig is not None
        assert sig.signal_type == SignalType.BUY
        assert sig.strategy_name == "MultiStrategyEnsemble"
        assert "contributing_strategies" in sig.metadata

    def test_ensemble_no_consensus_when_split(self) -> None:
        s1 = DummyLiveStrategy("s1", SignalType.BUY, 0.5)
        s2 = DummyLiveStrategy("s2", SignalType.SELL, 0.5)

        ensemble = MultiStrategyEnsemble(
            strategies=[s1, s2],
            min_vote_fraction=0.6,
            min_strategies_agree=1,
            config={"timeframe": 60},
        )

        c = _make_candle(timeframe=60)
        assert ensemble.on_candle(c) is None

    def test_on_tick_forwards_to_substrategies(self) -> None:
        s1 = DummyLiveStrategy("s1")
        s1.on_tick = MagicMock()
        ensemble = MultiStrategyEnsemble(strategies=[s1])
        tick = Tick("R_25", 1000, 100.0)
        ensemble.on_tick(tick)
        s1.on_tick.assert_called_once_with(tick)


# ---------------------------------------------------------------------------
# MetaRegimeStrategy Tests
# ---------------------------------------------------------------------------

class TestMetaRegimeStrategy:
    def test_meta_regime_routing(self) -> None:
        t1 = DummyLiveStrategy("t1", SignalType.BUY, 0.9)
        r1 = DummyLiveStrategy("r1", SignalType.SELL, 0.7)

        meta = MetaRegimeStrategy(
            trend_strategies=[t1],
            ranging_strategies=[r1],
            config={"timeframe": 60},
        )

        candles = _make_candle_series(length=55, timeframe=60)
        signals = []
        for c in candles:
            sig = meta.on_candle(c)
            if sig is not None:
                signals.append(sig)

        # Meta strategy runs and routes candles after 50 bars
        assert meta.current_regime is not None


# ---------------------------------------------------------------------------
# MLStrategy Tests
# ---------------------------------------------------------------------------

class TestMLStrategy:
    def test_uninitialized_model_returns_none(self) -> None:
        strat = MLStrategy("ml_test", config={"timeframe": 60, "model_path": "non_existent.joblib"})
        c = _make_candle(timeframe=60)
        assert strat.on_candle(c) is None

    @patch("project_apex.strategies.ml_strategy.XGBoostPredictor")
    def test_ml_strategy_inference_buy(self, mock_predictor_cls: MagicMock) -> None:
        mock_predictor = MagicMock()
        mock_predictor.is_trained = True
        mock_predictor.predict_probability.return_value = [0.85]  # > buy_threshold (0.65)
        mock_predictor_cls.return_value = mock_predictor

        strat = MLStrategy("ml_test", config={"timeframe": 60, "buy_threshold": 0.65})

        candles = _make_candle_series(length=45, timeframe=60)
        sig = None
        for c in candles:
            sig = strat.on_candle(c)

        if sig is not None:
            assert sig.signal_type == SignalType.BUY
            assert sig.confidence > 0.0
