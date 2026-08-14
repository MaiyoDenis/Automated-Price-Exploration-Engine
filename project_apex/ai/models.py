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

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """
        Trains the XGBoost model.

        If X_val/y_val are provided, early stopping is used to halt training
        when validation logloss stops improving — avoids overfitting and cuts
        unnecessary tree builds.

        Args:
            X_train: DataFrame of features.
            y_train: Series of labels (0 or 1).
            X_val:   Optional validation features for early stopping.
            y_val:   Optional validation labels for early stopping.
            params:  Optional dict of XGBoost hyperparameters.
        """
        default_params: dict[str, Any] = {
            "n_estimators": 300,        # higher ceiling — early stopping will cut it short
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "early_stopping_rounds": 15,  # stop if no improvement for 15 rounds
        }
        if params:
            default_params.update(params)

        logger.info(f"[XGBoostPredictor] Training model with {len(X_train)} samples...")
        self.model = xgb.XGBClassifier(**default_params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_kwargs)
        self.is_trained = True

        best_iter = getattr(self.model, "best_iteration", default_params["n_estimators"])
        logger.info(f"[XGBoostPredictor] Stopped at iteration {best_iter}")

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.success(f"[XGBoostPredictor] Training complete. Saved to {self.model_path}")

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of Class 1 (Price goes UP).

        When the model is not yet trained, returns a neutral 0.5 array so that
        callers receive a usable (low-confidence) probability rather than raising.

        Args:
            features: A DataFrame containing exactly the features the model was trained on.

        Returns:
            An array of probabilities [0.0, 1.0].
        """
        if not self.is_trained or self.model is None:
            logger.debug(
                "[XGBoostPredictor] Model not trained yet — returning neutral 0.5 probabilities."
            )
            return np.full(len(features), 0.5)

        # predict_proba returns [[prob_0, prob_1], ...]
        probas = self.model.predict_proba(features)
        return probas[:, 1]  # Return probability of Class 1

    def quick_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Compute a fast 2-fold cross-validated ROC-AUC on the given data.
        Used as a cheap gate before the full walk-forward Sharpe comparison.

        Returns AUC in [0, 1], or 0.5 on failure.
        """
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.base import clone

            base_params: dict[str, Any] = {
                "n_estimators": 50,   # intentionally small — this is just a gate
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            }
            probe = xgb.XGBClassifier(**base_params)
            scores = cross_val_score(probe, X, y, cv=2, scoring="roc_auc", n_jobs=1)
            return float(scores.mean())
        except Exception as exc:
            logger.warning(f"[XGBoostPredictor] quick_auc failed: {exc}")
            return 0.5

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Returns a dictionary of feature importances."""
        if not self.is_trained or self.model is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
