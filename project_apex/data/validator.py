"""
Project APEX
Data Validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from project_apex.models.tick import Tick
from project_apex.models.candle import Candle


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    rule: str | None = None
    field: str | None = None
    observed_value: Any | None = None


def validate_tick(tick: Tick, latest_timestamp: int | None) -> ValidationResult:
    """Validates a tick against data integrity rules."""
    
    if not tick.symbol:
        return ValidationResult(is_valid=False, rule="symbol_present", field="symbol", observed_value=tick.symbol)
        
    if tick.price <= 0:
        return ValidationResult(is_valid=False, rule="price_positive", field="price", observed_value=tick.price)
        
    if tick.timestamp <= 0:
        return ValidationResult(is_valid=False, rule="timestamp_positive", field="timestamp", observed_value=tick.timestamp)
        
    if latest_timestamp is not None and tick.timestamp < latest_timestamp:
        return ValidationResult(is_valid=False, rule="sequence", field="timestamp", observed_value=tick.timestamp)
        
    return ValidationResult(is_valid=True)


def validate_candle(candle: Candle, valid_timeframes: list[int]) -> ValidationResult:
    """Validates a candle against data integrity rules."""
    
    if candle.timeframe not in valid_timeframes:
        return ValidationResult(is_valid=False, rule="valid_timeframe", field="timeframe", observed_value=candle.timeframe)
        
    if candle.low > candle.high:
        return ValidationResult(is_valid=False, rule="ohlc_integrity", field="low/high", observed_value=(candle.low, candle.high))
        
    if not (candle.low <= candle.open <= candle.high):
        return ValidationResult(is_valid=False, rule="ohlc_integrity", field="open", observed_value=candle.open)
        
    if not (candle.low <= candle.close <= candle.high):
        return ValidationResult(is_valid=False, rule="ohlc_integrity", field="close", observed_value=candle.close)
        
    return ValidationResult(is_valid=True)
