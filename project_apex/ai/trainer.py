"""
Project APEX — Autonomous ML Trainer

Schedules automatic retraining of the XGBoost predictor on two triggers:
  1. Every 24 hours (scheduled)
  2. On feature drift detection — when the distribution of ML input features
     (RSI, MACD, BB width, ADX, etc.) has shifted significantly from the
     training baseline, measured via Population Stability Index (PSI).

Walk-forward validation: The model is ONLY saved if the new model achieves
a better Sharpe ratio on the holdout validation split than the current model.
This prevents replacing a good model with an overfit one.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from project_apex.ai.models import XGBoostPredictor
from project_apex.ai.features import FeatureGenerator
from project_apex.database.sqlite_manager import SQLiteManager


# Minimum samples required before retraining is attempted
_MIN_TRAINING_ROWS = 2000

# Holdout fraction for walk-forward validation
_VALIDATION_SPLIT = 0.20

# Minimum Sharpe improvement required to replace the current model
_MIN_SHARPE_IMPROVEMENT = 0.05

# PSI thresholds
# PSI < 0.10  → no significant change
# PSI 0.10–0.25 → moderate shift (monitor)
# PSI > 0.25  → significant drift → trigger retrain
_PSI_BINS = 10
_PSI_DRIFT_THRESHOLD = 0.25


class ModelTrainer:
    """
    Autonomous model retraining scheduler.

    Args:
        predictor: The live XGBoostPredictor instance used for inference.
        db: The SQLiteManager providing access to historical candles.
        symbol: Primary symbol used for training data.
        timeframe: Candle timeframe (seconds) for training.
        forward_bars: How many bars forward to use as the prediction target.
        lookback_days: How many days of history to use in each training run.
        retrain_interval_h: Scheduled retraining interval in hours.
        drift_check_interval_m: How often (minutes) to check for regime drift.
    """

    def __init__(
        self,
        predictor: XGBoostPredictor,
        db: SQLiteManager,
        symbol: str = "R_50",
        timeframe: int = 300,
        forward_bars: int = 3,
        lookback_days: int = 30,
        retrain_interval_h: float = 24.0,
        drift_check_interval_m: float = 60.0,
    ) -> None:
        self._predictor = predictor
        self._db = db
        self._symbol = symbol
        self._timeframe = timeframe
        self._forward_bars = forward_bars
        self._lookback_days = lookback_days
        self._retrain_interval_s = retrain_interval_h * 3600
        self._drift_check_interval_s = drift_check_interval_m * 60
        self._feature_gen = FeatureGenerator()

        self._scheduled_task: Optional[asyncio.Task] = None
        self._drift_task: Optional[asyncio.Task] = None
        self._running = False

        # Feature distribution baselines captured at training time (PSI drift detection).
        # Maps feature_name -> (bin_edges np.ndarray, expected_pct np.ndarray)
        self._training_feature_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        logger.info(
            f"[ModelTrainer] Initialized | symbol={symbol} | "
            f"retrain_every={retrain_interval_h}h | "
            f"drift_check_every={drift_check_interval_m}m"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start scheduled retraining and drift detection tasks."""
        self._running = True
        self._scheduled_task = asyncio.create_task(self._scheduled_loop())
        self._drift_task = asyncio.create_task(self._drift_detection_loop())
        logger.info("[ModelTrainer] Autonomous training tasks started.")

    async def stop(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in [self._scheduled_task, self._drift_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("[ModelTrainer] Stopped.")

    # ── Retraining entry point ────────────────────────────────────────────────

    async def retrain(self, trigger: str = "scheduled") -> bool:
        """
        Run a full retrain cycle.

        Args:
            trigger: Why the retrain was triggered (for logging).

        Returns:
            True if the model was updated, False otherwise.
        """
        logger.info(f"[ModelTrainer] Retraining triggered by: {trigger}")

        # Run heavy computation in a thread pool to avoid blocking the event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._retrain_sync, trigger
        )
        return result

    def _retrain_sync(self, trigger: str) -> bool:
        """Blocking retrain — runs in executor."""
        try:
            df = self._load_data()
            if df is None or len(df) < _MIN_TRAINING_ROWS:
                logger.warning(
                    f"[ModelTrainer] Not enough data ({len(df) if df is not None else 0} rows, "
                    f"need {_MIN_TRAINING_ROWS}). Skipping retrain."
                )
                return False

            # Generate features
            df_feat = self._feature_gen.generate(df)
            feature_cols = self._feature_gen.get_feature_columns()

            # Create labels: 1 if close N bars forward > close now, else 0
            df_feat["label"] = (
                df_feat["close"].shift(-self._forward_bars) > df_feat["close"]
            ).astype(int)

            # Drop rows with NaN in features or label
            df_clean = df_feat.dropna(subset=feature_cols + ["label"])
            if len(df_clean) < _MIN_TRAINING_ROWS:
                logger.warning("[ModelTrainer] Too many NaN rows after feature generation. Skipping.")
                return False

            # Walk-forward split (train on earlier data, validate on recent)
            split_idx = int(len(df_clean) * (1 - _VALIDATION_SPLIT))
            train_df = df_clean.iloc[:split_idx]
            val_df = df_clean.iloc[split_idx:]

            X_train = train_df[feature_cols]
            y_train = train_df["label"]
            X_val = val_df[feature_cols]
            y_val = val_df["label"]

            # Store per-feature distribution baselines for PSI drift detection.
            # We record the expected bucket percentages over the training set.
            self._training_feature_baselines = self._compute_feature_baselines(
                df_feat, feature_cols
            )

            # Evaluate current model on validation set
            current_sharpe = self._evaluate_model(self._predictor, X_val, y_val)

            # Train a new candidate model
            new_predictor = XGBoostPredictor.__new__(XGBoostPredictor)
            new_predictor.model = None
            new_predictor.is_trained = False
            new_predictor.model_path = self._predictor.model_path

            new_predictor.train(X_train, y_train)
            new_sharpe = self._evaluate_model(new_predictor, X_val, y_val)

            logger.info(
                f"[ModelTrainer] Validation Sharpe — current={current_sharpe:.3f} | "
                f"new={new_sharpe:.3f} | trigger={trigger}"
            )

            if new_sharpe >= current_sharpe + _MIN_SHARPE_IMPROVEMENT:
                # Replace the live model in-place
                self._predictor.model = new_predictor.model
                self._predictor.is_trained = True
                import joblib
                import os
                os.makedirs(os.path.dirname(self._predictor.model_path), exist_ok=True)
                joblib.dump(self._predictor.model, self._predictor.model_path)
                logger.success(
                    f"[ModelTrainer] ✅ Model updated! Sharpe: {current_sharpe:.3f} → "
                    f"{new_sharpe:.3f} (trigger={trigger})"
                )
                return True
            else:
                logger.info(
                    f"[ModelTrainer] Model NOT updated — new model Sharpe "
                    f"({new_sharpe:.3f}) did not beat current ({current_sharpe:.3f}) "
                    f"by {_MIN_SHARPE_IMPROVEMENT} threshold."
                )
                return False

        except Exception as exc:
            logger.error(f"[ModelTrainer] Retrain failed: {exc}")
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_data(self) -> Optional[pd.DataFrame]:
        """Load recent candles from SQLite."""
        import time
        lookback_s = self._lookback_days * 86400
        start_ts = int(time.time()) - lookback_s

        query = """
        SELECT timestamp, open, high, low, close
        FROM candles
        WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
        ORDER BY timestamp ASC
        """
        try:
            rows = self._db.fetchall(query, (self._symbol, self._timeframe, start_ts))
            if not rows:
                return None
            return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
        except Exception as exc:
            logger.error(f"[ModelTrainer] Data load error: {exc}")
            return None

    def _evaluate_model(self, predictor: XGBoostPredictor, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Evaluate model performance on validation data.
        Returns an approximate Sharpe using signal-based returns.
        """
        if not predictor.is_trained:
            return -999.0
        try:
            probs = predictor.predict_probability(X_val)
            # Simple signal: buy when prob > 0.55, sell when prob < 0.45
            signals = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
            # Strategy returns: signal * actual next-bar direction
            actual_direction = (y_val.values * 2) - 1  # {0,1} → {-1, 1}
            strategy_returns = signals * actual_direction
            if len(strategy_returns) < 5:
                return 0.0
            mean_r = np.mean(strategy_returns)
            std_r = np.std(strategy_returns)
            return float(mean_r / std_r) if std_r > 0 else 0.0
        except Exception:
            return -999.0

    async def _scheduled_loop(self) -> None:
        """Trigger a retrain every retrain_interval_s seconds."""
        while self._running:
            await asyncio.sleep(self._retrain_interval_s)
            try:
                await self.retrain(trigger="24h_schedule")
            except Exception as exc:
                logger.error(f"[ModelTrainer] Scheduled retrain error: {exc}")

    async def _drift_detection_loop(self) -> None:
        """Periodically check if the market has drifted from the training distribution."""
        while self._running:
            await asyncio.sleep(self._drift_check_interval_s)
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._check_drift
                )
            except Exception as exc:
                logger.error(f"[ModelTrainer] Drift check error: {exc}")

    # ── PSI Drift Detection ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_feature_baselines(
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute bin edges and expected percentages for each feature column.
        Returns a dict of {feature_name: (bin_edges, expected_pct)}.
        """
        baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col in feature_cols:
            series = df[col].dropna().values
            if len(series) < _PSI_BINS * 2:
                continue
            counts, edges = np.histogram(series, bins=_PSI_BINS)
            total = counts.sum()
            expected_pct = (counts / total).clip(1e-6)  # avoid log(0)
            baselines[col] = (edges, expected_pct)
        return baselines

    @staticmethod
    def _calculate_psi(
        expected_pct: np.ndarray,
        bin_edges: np.ndarray,
        actual_values: np.ndarray,
    ) -> float:
        """
        Population Stability Index between two distributions.

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Bins are defined by ``bin_edges`` from the training data.
        """
        actual_counts, _ = np.histogram(actual_values, bins=bin_edges)
        total = actual_counts.sum()
        if total == 0:
            return 0.0
        actual_pct = (actual_counts / total).clip(1e-6)
        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return max(0.0, psi)

    def _check_drift(self) -> None:
        """
        Population Stability Index (PSI) drift check.

        For each ML input feature (RSI, MACD histogram, BB width, ADX, etc.),
        compute the PSI between the training-time distribution and the
        distribution of the most recent 200 candles.

        If any feature exceeds the PSI threshold (> 0.25) an immediate retrain
        is triggered.
        """
        if not self._training_feature_baselines:
            return

        df = self._load_data()
        if df is None or len(df) < 200:
            return

        # Generate features for the recent window
        try:
            df_recent = self._feature_gen.generate(df.tail(400))
        except Exception as exc:
            logger.error(f"[ModelTrainer] Feature generation failed during drift check: {exc}")
            return

        drifted_features: list[tuple[str, float]] = []
        for feature_name, (bin_edges, expected_pct) in self._training_feature_baselines.items():
            actual_values = df_recent[feature_name].dropna().values
            if len(actual_values) < _PSI_BINS * 2:
                continue
            psi = self._calculate_psi(expected_pct, bin_edges, actual_values)
            logger.debug(
                f"[ModelTrainer] PSI drift check — feature={feature_name} | PSI={psi:.4f}"
            )
            if psi > _PSI_DRIFT_THRESHOLD:
                drifted_features.append((feature_name, psi))

        if drifted_features:
            worst = max(drifted_features, key=lambda x: x[1])
            feature_summary = ", ".join(
                f"{name}(PSI={psi:.3f})" for name, psi in drifted_features
            )
            logger.warning(
                f"[ModelTrainer] ⚠ Feature drift detected! "
                f"Drifted features: [{feature_summary}] — "
                f"worst: {worst[0]}={worst[1]:.3f} (threshold={_PSI_DRIFT_THRESHOLD}) — "
                f"triggering immediate retrain."
            )
            asyncio.run_coroutine_threadsafe(
                self.retrain(
                    trigger=f"psi_drift({worst[0]}_psi={worst[1]:.3f})"
                ),
                asyncio.get_event_loop(),
            )
        else:
            logger.debug(
                f"[ModelTrainer] Drift check PASSED — all {len(self._training_feature_baselines)} "
                f"features stable (PSI ≤ {_PSI_DRIFT_THRESHOLD})"
            )
