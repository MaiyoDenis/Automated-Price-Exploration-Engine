"""
Project APEX — ML Strategy

A live strategy that uses the XGBoostPredictor to emit TradeSignals.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
import pandas as pd

from project_apex.models.candle import Candle
from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.ai.features import FeatureGenerator
from project_apex.ai.models import XGBoostPredictor


class MLStrategy(LiveStrategy):
    """
    Predictive strategy powered by XGBoost.
    
    Generates features for every new candle and passes them to the ML model.
    Emits BUY if the probability of the price going UP > buy_threshold.
    Emits SELL if the probability of the price going UP < sell_threshold (meaning DOWN is highly probable).
    """

    # We need enough history to calculate the longest indicator (e.g. MACD slow=26)
    _MIN_BARS = 40

    def initialize(self, config: dict[str, Any]) -> None:
        self._timeframe: int = config.get("timeframe", 60)
        self._buy_threshold: float = config.get("buy_threshold", 0.65)
        self._sell_threshold: float = config.get("sell_threshold", 0.35)
        model_path: str = config.get("model_path", "datasets/xgb_model.joblib")

        self.feature_gen = FeatureGenerator()
        self.predictor = XGBoostPredictor(model_path=model_path)
        self.feature_columns = self.feature_gen.get_feature_columns()

        logger.info(
            f"[{self.name}] MLStrategy initialized | "
            f"buy_prob > {self._buy_threshold:.2f} | sell_prob < {self._sell_threshold:.2f} | "
            f"model_trained={self.predictor.is_trained}"
        )

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        if candle.timeframe != self._timeframe:
            return None
            
        if not self.predictor.is_trained:
            return None  # Cannot trade without a trained model

        history = self._append_candle(candle)
        if len(history) < self._MIN_BARS:
            return None

        # 1. Prepare history dataframe
        df = self._get_history_df(candle)
        
        # 2. Generate features
        df_features = self.feature_gen.generate(df)
        
        # 3. Extract the latest row's features
        latest_row = df_features.iloc[-1:]
        X = latest_row[self.feature_columns]
        
        # Guard against NaNs in features (XGBoost can handle them sometimes, but safer to skip)
        if X.isna().any().any():
            return None

        # 4. Predict probability
        try:
            prob_up = self.predictor.predict_probability(X)[0]
        except Exception as e:
            logger.error(f"[{self.name}] Inference error: {e}")
            return None

        # 5. Threshold logic
        signal_type: SignalType | None = None
        confidence: float = 0.0

        if prob_up > self._buy_threshold:
            signal_type = SignalType.BUY
            confidence = (prob_up - self._buy_threshold) / (1.0 - self._buy_threshold)
            
        elif prob_up < self._sell_threshold:
            signal_type = SignalType.SELL
            confidence = (self._sell_threshold - prob_up) / self._sell_threshold

        if signal_type is None:
            return None

        confidence = max(0.1, min(1.0, confidence))

        logger.debug(
            f"[{self.name}] {signal_type.name} on {candle.symbol} | "
            f"Prob(UP)={prob_up:.2%} conf={confidence:.2f}"
        )

        return TradeSignal(
            symbol=candle.symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            metadata={
                "prob_up": float(prob_up),
                "timeframe": candle.timeframe,
            },
        )
