"""
Project APEX — VWAP Scalper Strategy

A fast-paced scalping strategy that uses Volume Weighted Average Price (VWAP)
combined with an Exponential Moving Average (EMA) crossover to capture
short-term momentum bursts.
"""
from typing import Any
import pandas as pd
from loguru import logger

from project_apex.strategies.base import LiveStrategy
from project_apex.strategies.signals import TradeSignal, SignalType
from project_apex.models.candle import Candle
from project_apex.indicators.volume import VWAP
from project_apex.indicators.moving_averages import EMA


class VWAPScalperStrategy(LiveStrategy):
    """
    VWAP + EMA Scalping Strategy.
    
    Rules:
    - BUY: Price is above VWAP (bullish context) AND Fast EMA (9) crosses above Slow EMA (21).
    - SELL: Price is below VWAP (bearish context) AND Fast EMA (9) crosses below Slow EMA (21).
    """

    def initialize(self, config: dict[str, Any]) -> None:
        self.vwap = VWAP(period=20)
        self.ema_fast = EMA(period=9)
        self.ema_slow = EMA(period=21)
        self.target_timeframe = config.get("timeframe", 60)
        logger.info(f"[{self.name}] Initialized | VWAP(20) EMA(9,21) timeframe={self.target_timeframe}s")

    def on_candle(self, candle: Candle) -> TradeSignal | None:
        # Only process candles on the fast timeframe
        if candle.timeframe != self.target_timeframe:
            return None

        self._append_candle(candle)
        df = self._get_history_df(candle)
        
        # Need enough history for VWAP and EMAs.
        # Minimum is 22: EMA(21) uses ewm so it starts from candle 1,
        # but we need at least a prev + latest pair after period-21 settles.
        # The NaN guard below handles any remaining edge cases.
        if len(df) < 22:
            return TradeSignal(
                symbol=candle.symbol,
                signal_type=SignalType.HOLD,
                price=candle.close,
                timestamp=candle.timestamp,
                strategy_name=self.name,
                confidence=0.0
            )
            
        # The volume indicators expect a 'volume' column, but our history has 'tick_count'
        df = df.rename(columns={"tick_count": "volume"})
        
        df = self.vwap.calculate(df)
        df = self.ema_fast.calculate(df)
        df = self.ema_slow.calculate(df)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for NaN in calculated indicators before proceeding
        if pd.isna(latest["vwap_20"]) or pd.isna(latest["ema_9"]) or pd.isna(latest["ema_21"]):
            return TradeSignal(
                symbol=candle.symbol,
                signal_type=SignalType.HOLD,
                price=candle.close,
                timestamp=candle.timestamp,
                strategy_name=self.name,
                confidence=0.0
            )

        # Bullish Scalp Signal
        if latest["close"] > latest["vwap_20"]:
            if prev["ema_9"] <= prev["ema_21"] and latest["ema_9"] > latest["ema_21"]:
                return TradeSignal(
                    symbol=candle.symbol,
                    signal_type=SignalType.BUY,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    strategy_name=self.name,
                    confidence=0.85,
                    metadata={
                        "vwap": round(latest["vwap_20"], 5), 
                        "ema9": round(latest["ema_9"], 5), 
                        "ema21": round(latest["ema_21"], 5)
                    }
                )
                
        # Bearish Scalp Signal
        if latest["close"] < latest["vwap_20"]:
            if prev["ema_9"] >= prev["ema_21"] and latest["ema_9"] < latest["ema_21"]:
                return TradeSignal(
                    symbol=candle.symbol,
                    signal_type=SignalType.SELL,
                    price=candle.close,
                    timestamp=candle.timestamp,
                    strategy_name=self.name,
                    confidence=0.85,
                    metadata={
                        "vwap": round(latest["vwap_20"], 5), 
                        "ema9": round(latest["ema_9"], 5), 
                        "ema21": round(latest["ema_21"], 5)
                    }
                )
                
        return TradeSignal(
            symbol=candle.symbol,
            signal_type=SignalType.HOLD,
            price=candle.close,
            timestamp=candle.timestamp,
            strategy_name=self.name,
            confidence=0.0
        )
