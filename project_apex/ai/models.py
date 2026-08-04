"""
Project APEX — AI Predictive Models

Wraps XGBoost for predicting trade win probabilities based on technical features.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from loguru import logger


class XGBoostPredictor:
    """
    XGBoost classification model.
    Predicts the probability of the price going UP (Class 1) or DOWN (Class 0)
    in the next N candles.
    """

    def __init__(self, model_path: str = "datasets/xgb_model.joblib") -> None:
        self.model_path = model_path
        self.model: xgb.XGBClassifier | None = None
        self.is_trained = False
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        """Loads the model from disk if available."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"[XGBoostPredictor] Loaded model from {self.model_path}")
            except Exception as e:
                logger.error(f"[XGBoostPredictor] Failed to load model: {e}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any] | None = None) -> None:
        """
        Trains the XGBoost model.
        
        Args:
            X_train: DataFrame of features.
            y_train: Series of labels (0 or 1).
            params: Optional dict of XGBoost hyperparameters.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42
        }
        if params:
            default_params.update(params)

        logger.info(f"[XGBoostPredictor] Training model with {len(X_train)} samples...")
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.success(f"[XGBoostPredictor] Training complete. Saved to {self.model_path}")

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of Class 1 (Price goes UP).
        
        Args:
            features: A DataFrame containing exactly the features the model was trained on.
            
        Returns:
            An array of probabilities [0.0, 1.0].
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() first or load an existing model.")
        
        # predict_proba returns [[prob_0, prob_1], ...]
        probas = self.model.predict_proba(features)
        return probas[:, 1]  # Return probability of Class 1

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Returns a dictionary of feature importances."""
        if not self.is_trained or self.model is None:
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
