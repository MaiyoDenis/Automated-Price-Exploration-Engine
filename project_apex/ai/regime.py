"""
Project APEX — Market Regime Detection

Analyzes recent price action to classify the current market environment:
- STRONG_TREND_UP
- STRONG_TREND_DOWN
- RANGING (Choppy)
- HIGH_VOLATILITY
"""
from __future__ import annotations

from enum import Enum
import pandas as pd
import numpy as np


class MarketRegime(Enum):
    STRONG_TREND_UP = "TREND_UP"
    STRONG_TREND_DOWN = "TREND_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "VOLATILE"


class RegimeDetector:
    """
    Detects the market regime using ADX (trend strength) and ATR (volatility).
    """

    def __init__(self, adx_threshold: float = 25.0, volatility_percentile: float = 0.8) -> None:
        """
        Args:
            adx_threshold: ADX > threshold indicates a strong trend.
            volatility_percentile: If current ATR is above the historical Nth percentile, 
                                   it's considered HIGH_VOLATILITY.
        """
        self.adx_threshold = adx_threshold
        self.vol_percentile = volatility_percentile

    def detect(self, df: pd.DataFrame, adx_col: str, plus_di_col: str, minus_di_col: str, atr_col: str) -> MarketRegime:
        """
        Evaluates the latest row in the DataFrame to determine the regime.
        """
        if len(df) < 50:
            return MarketRegime.RANGING  # Default if not enough data
            
        latest = df.iloc[-1]
        
        adx = latest.get(adx_col, np.nan)
        plus_di = latest.get(plus_di_col, np.nan)
        minus_di = latest.get(minus_di_col, np.nan)
        atr = latest.get(atr_col, np.nan)
        
        if pd.isna(adx) or pd.isna(atr):
            return MarketRegime.RANGING

        # 1. Check Volatility Spike
        # Get the historical 80th percentile of ATR
        recent_atr = df[atr_col].tail(200)
        high_vol_threshold = recent_atr.quantile(self.vol_percentile)
        
        if atr > high_vol_threshold:
            # Extreme volatility supersedes normal trend rules (often unpredictable)
            return MarketRegime.HIGH_VOLATILITY

        # 2. Check Trend
        if adx > self.adx_threshold:
            if plus_di > minus_di:
                return MarketRegime.STRONG_TREND_UP
            else:
                return MarketRegime.STRONG_TREND_DOWN
                
        # 3. Default
        return MarketRegime.RANGING
