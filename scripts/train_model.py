"""
Script to train the XGBoost Predictor for Project APEX.

Usage:
    python scripts/train_model.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from loguru import logger

from project_apex.database.sqlite_manager import SQLiteManager
from project_apex.ai.features import FeatureGenerator
from project_apex.ai.models import XGBoostPredictor
from project_apex.config.config import Config


def load_data(db_path: str, symbol: str, timeframe: int) -> pd.DataFrame:
    db = SQLiteManager(db_path)
    db.connect()
    
    query = """
    SELECT timestamp, open, high, low, close 
    FROM candles 
    WHERE symbol = ? AND timeframe = ?
    ORDER BY timestamp ASC
    """
    rows = db.fetchall(query, (symbol, timeframe))
    db.close()
    
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close'])
    return df


def create_targets(df: pd.DataFrame, lookahead: int = 3) -> pd.DataFrame:
    """
    Creates target labels for the ML model.
    1 if the close price `lookahead` candles in the future is higher than current close, else 0.
    """
    # Shift the close price backwards to align future price with current row
    future_close = df['close'].shift(-lookahead)
    
    # 1 if future price is higher, 0 if lower or equal
    df['target'] = np.where(future_close > df['close'], 1, 0)
    
    # The last `lookahead` rows will have NaN targets, we must drop them
    # but we'll let the caller dropna on the final dataframe
    df.loc[df.index[-lookahead:], 'target'] = np.nan
    return df


def main():
    config = Config()
    db_path = config.get_str("database", "path")
    symbol = "R_25"
    timeframe = 60
    
    logger.info(f"Loading data for {symbol} ({timeframe}s) from {db_path}...")
    df = load_data(db_path, symbol, timeframe)
    
    if df.empty:
        logger.error("No data found! Please run the collector to gather some market data first.")
        return
        
    logger.info(f"Loaded {len(df)} candles. Generating features...")
    
    # 1. Generate Features
    feature_gen = FeatureGenerator()
    df = feature_gen.generate(df)
    
    # 2. Create Targets
    df = create_targets(df, lookahead=5)  # Predict 5 minutes out
    
    # 3. Clean Data
    df_clean = df.dropna().copy()
    logger.info(f"Data cleaned. {len(df_clean)} samples remaining for training.")
    
    if len(df_clean) < 100:
        logger.error("Not enough data to train. Need at least a few hundred rows.")
        return

    # 4. Train/Test Split (Temporal)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    feature_cols = feature_gen.get_feature_columns()
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    # 5. Train Model
    predictor = XGBoostPredictor()
    predictor.train(X_train, y_train)
    
    # 6. Basic Evaluation
    from sklearn.metrics import accuracy_score, classification_report
    
    y_pred_prob = predictor.predict_probability(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    logger.success(f"Model Accuracy on unseen test data: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    
    # 7. Feature Importance
    importances = predictor.get_feature_importance(feature_cols)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Important Features:")
    for f, imp in sorted_importances[:5]:
        print(f"  {f}: {imp:.4f}")


if __name__ == "__main__":
    main()
