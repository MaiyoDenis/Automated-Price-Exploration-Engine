"""
Project APEX — Autonomous ML Trainer

Schedules automatic retraining of the XGBoost predictor on two triggers:
  1. Every 24 hours (scheduled)
  2. On feature drift detection — when the distribution of ML input features
     (RSI, MACD, BB width, ADX, etc.) has shifted significantly from the
     training baseline, measured via Population Stability Index (PSI).

Walk-forward validation with a cheap AUC gate:
  - A quick 2-fold AUC check is run first. If AUC < _MIN_AUC_GATE the
    retrain is aborted early without paying the full walk-forward cost.
  - Only when AUC passes is the full Sharpe comparison done.
  - The model is saved only if the new Sharpe beats the current model by
    at least _MIN_SHARPE_IMPROVEMENT.

Performance improvements over v1:
  - Cold-start floor lowered from 2000 → 500 rows so the model comes
    online much faster (≈1.7 days at 5m TF instead of ≈7 days).
  - predict_probability() returns neutral 0.5 when untrained instead
    of raising, so inference never blocks.
  - XGBoost uses early_stopping_rounds=15 — avoids training all 300
    trees when the model converges earlier.
  - _load_data() accepts an optional rows_limit so drift checks only
    fetch the last ~600 rows instead of a full 30-day dataset.
  - FeatureGenerator caches results — drift checks that run on
    overlapping data reuse the cached feature DataFrame.
  - _check_drift() receives the running event loop at construction time
    to safely schedule coroutines from the executor thread.
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


# Minimum samples required before retraining is attempted.
# Lowered from 2000 → 500: at a 5-minute timeframe this means training
# starts after ~1.7 days of data instead of ~7 days.
_MIN_TRAINING_ROWS = 500

# Holdout fraction for walk-forward validation
_VALIDATION_SPLIT = 0.20

# Cheap AUC gate: if a quick 2-fold AUC on the candidate is below this
# value, skip the full walk-forward Sharpe comparison and abort.
_MIN_AUC_GATE = 0.52

# Minimum Sharpe improvement required to replace the current model
_MIN_SHARPE_IMPROVEMENT = 0.05

# PSI thresholds
# PSI < 0.10  → no significant change
# PSI 0.10–0.25 → moderate shift (monitor)
# PSI > 0.25  → significant drift → trigger retrain
_PSI_BINS = 10
_PSI_DRIFT_THRESHOLD = 0.25

# Number of recent rows fetched for drift checks.
# Much smaller than the full 30-day training window — drift only needs
# a recent snapshot, not the entire history.
_DRIFT_FETCH_ROWS = 600


class ModelTrainer:
    """
    Autonomous model retraining scheduler.

    Args:
        predictor:              The live XGBoostPredictor instance used for inference.
        db:                     The SQLiteManager providing access to historical candles.
        symbol:                 Primary symbol used for training data.
        timeframe:              Candle timeframe (seconds) for training.
        forward_bars:           How many bars forward to use as the prediction target.
        lookback_days:          How many days of history to use in each training run.
        retrain_interval_h:     Scheduled retraining interval in hours.
        drift_check_interval_m: How often (minutes) to check for regime drift.
        loop:                   The running asyncio event loop. Must be supplied so
                                that _check_drift (runs in a thread) can safely
                                schedule coroutines back onto it.
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
        loop: asyncio.AbstractEventLoop | None = None,
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

        # Store the event loop explicitly so executor threads can safely
        # use asyncio.run_coroutine_threadsafe without hitting
        # "no running event loop" in Python 3.10+.
        self._loop: asyncio.AbstractEventLoop | None = loop

        self._scheduled_task: Optional[asyncio.Task] = None
        self._drift_task: Optional[asyncio.Task] = None
        self._running = False

        # Feature distribution baselines captured at training time (PSI drift detection).
        # Maps feature_name -> (bin_edges np.ndarray, expected_pct np.ndarray)
        self._training_feature_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        logger.info(
            f"[ModelTrainer] Initialized | symbol={symbol} | "
            f"min_rows={_MIN_TRAINING_ROWS} | "
            f"retrain_every={retrain_interval_h}h | "
            f"drift_check_every={drift_check_interval_m}m"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start scheduled retraining and drift detection tasks."""
        self._running = True
        # Capture the running loop so executor threads can schedule back onto it.
        self._loop = asyncio.get_running_loop()
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
                    f"[ModelTrainer] Not enough data "
                    f"({len(df) if df is not None else 0} rows, need {_MIN_TRAINING_ROWS}). "
                    f"Skipping retrain."
                )
                return False

            # Generate features — cache helps if drift triggered this shortly
            # after the last run on the same dataset.
            df_feat = self._feature_gen.generate(df)
            feature_cols = self._feature_gen.get_feature_columns()

            # Create labels: 1 if close N bars forward > close now, else 0
            df_feat = df_feat.copy()
            df_feat["label"] = (
                df_feat["close"].shift(-self._forward_bars) > df_feat["close"]
            ).astype(int)

            # Drop rows with NaN in features or label
            df_clean = df_feat.dropna(subset=feature_cols + ["label"])
            if len(df_clean) < _MIN_TRAINING_ROWS:
                logger.warning(
                    "[ModelTrainer] Too many NaN rows after feature generation. Skipping."
                )
                return False

            # Walk-forward split (train on earlier data, validate on recent)
            split_idx = int(len(df_clean) * (1 - _VALIDATION_SPLIT))
            train_df = df_clean.iloc[:split_idx]
            val_df = df_clean.iloc[split_idx:]

            X_train = train_df[feature_cols]
            y_train = train_df["label"]
            X_val = val_df[feature_cols]
            y_val = val_df["label"]

            # ── Gate 1: cheap AUC check ───────────────────────────────────
            # Build a tiny probe model (50 trees, 2-fold CV) to assess whether
            # the data has any signal at all before paying the full training cost.
            probe_auc = XGBoostPredictor.__new__(XGBoostPredictor)
            probe_auc.model = None
            probe_auc.is_trained = False
            probe_auc.model_path = self._predictor.model_path
            auc = probe_auc.quick_auc(X_train, y_train)
            logger.info(f"[ModelTrainer] AUC gate: {auc:.4f} (threshold={_MIN_AUC_GATE})")
            if auc < _MIN_AUC_GATE:
                logger.info(
                    f"[ModelTrainer] AUC gate NOT passed ({auc:.4f} < {_MIN_AUC_GATE}). "
                    f"Aborting retrain — data lacks sufficient signal."
                )
                return False

            # ── Store feature distribution baselines for PSI ──────────────
            self._training_feature_baselines = self._compute_feature_baselines(
                df_feat, feature_cols
            )
            # Invalidate cache so the next drift check recomputes on fresh data
            self._feature_gen.invalidate_cache()

            # ── Gate 2: walk-forward Sharpe comparison ────────────────────
            current_sharpe = self._evaluate_model(self._predictor, X_val, y_val)

            # Train the candidate model — pass val set for early stopping
            new_predictor = XGBoostPredictor.__new__(XGBoostPredictor)
            new_predictor.model = None
            new_predictor.is_trained = False
            new_predictor.model_path = self._predictor.model_path

            new_predictor.train(X_train, y_train, X_val=X_val, y_val=y_val)
            new_sharpe = self._evaluate_model(new_predictor, X_val, y_val)

            logger.info(
                f"[ModelTrainer] Validation Sharpe — current={current_sharpe:.3f} | "
                f"new={new_sharpe:.3f} | trigger={trigger}"
            )

            if new_sharpe >= current_sharpe + _MIN_SHARPE_IMPROVEMENT:
                # Replace the live model in-place (thread-safe: we only swap the
                # reference once training is complete)
                self._predictor.model = new_predictor.model
                self._predictor.is_trained = True
                import joblib
                import os
                os.makedirs(os.path.dirname(self._predictor.model_path), exist_ok=True)
                joblib.dump(self._predictor.model, self._predictor.model_path)
                logger.success(
                    f"[ModelTrainer] ✅ Model updated! "
                    f"Sharpe: {current_sharpe:.3f} → {new_sharpe:.3f} (trigger={trigger})"
                )
                return True
            else:
                logger.info(
                    f"[ModelTrainer] Model NOT updated — new Sharpe "
                    f"({new_sharpe:.3f}) did not beat current ({current_sharpe:.3f}) "
                    f"by {_MIN_SHARPE_IMPROVEMENT} threshold."
                )
                return False

        except Exception as exc:
            logger.error(f"[ModelTrainer] Retrain failed: {exc}", exc_info=True)
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_data(self, rows_limit: int | None = None) -> Optional[pd.DataFrame]:
        """
        Load recent candles from SQLite.

        Args:
            rows_limit: When set, only the most recent N rows are fetched.
                        Used by the drift check to avoid pulling the entire
                        30-day history when only a recent snapshot is needed.
        """
        import time
        lookback_s = self._lookback_days * 86400
        start_ts = int(time.time()) - lookback_s

        if rows_limit is not None:
            query = """
            SELECT timestamp, open, high, low, close
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            params = (self._symbol, self._timeframe, start_ts, rows_limit)
        else:
            query = """
            SELECT timestamp, open, high, low, close
            FROM candles
            WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """
            params = (self._symbol, self._timeframe, start_ts)

        try:
            rows = self._db.fetchall(query, params)
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
            # When using DESC + LIMIT the rows come back newest-first; reverse them.
            if rows_limit is not None:
                df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as exc:
            logger.error(f"[ModelTrainer] Data load error: {exc}")
            return None

    def _evaluate_model(
        self,
        predictor: XGBoostPredictor,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
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

    # ── PSI Drift Detection ───────────────────────────────────────────────────

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

        Fetches only the last _DRIFT_FETCH_ROWS candles (instead of the full
        30-day training window) for a fast, lightweight check.

        If any feature exceeds the PSI threshold (> 0.25) an immediate retrain
        is triggered via the stored event loop reference.
        """
        if not self._training_feature_baselines:
            return

        # Use a small scoped fetch — we only need a recent snapshot
        df = self._load_data(rows_limit=_DRIFT_FETCH_ROWS)
        if df is None or len(df) < 200:
            return

        # Feature generation will hit the cache if this window overlaps
        # the tail of the last full training run
        try:
            df_recent = self._feature_gen.generate(df)
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

            # Use the stored event loop to safely schedule the coroutine from
            # this executor thread. Avoids the Python 3.10+ issue where
            # asyncio.get_event_loop() from a thread returns a new loop.
            if self._loop is not None and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.retrain(
                        trigger=f"psi_drift({worst[0]}_psi={worst[1]:.3f})"
                    ),
                    self._loop,
                )
            else:
                logger.error(
                    "[ModelTrainer] Cannot schedule drift-triggered retrain — "
                    "event loop not available."
                )
        else:
            logger.debug(
                f"[ModelTrainer] Drift check PASSED — all "
                f"{len(self._training_feature_baselines)} features stable "
                f"(PSI ≤ {_PSI_DRIFT_THRESHOLD})"
            )
