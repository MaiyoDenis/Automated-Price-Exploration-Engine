"""
Project APEX
Differs Signal

Extends TradeSignal for digit prediction contracts (DIGITDIFF / DIGITMATCH).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from project_apex.strategies.signals import TradeSignal, SignalType


@dataclass(frozen=True)
class DifferSignal(TradeSignal):
    """
    A signal specifically for digit contracts.
    
    Attributes:
        barrier: The digit to predict against (0-9). For Differs, we predict the last digit WILL NOT be this.
        duration_ticks: How many ticks the contract runs for.
    """
    barrier: int = 0
    duration_ticks: int = 1
    
    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0 <= self.barrier <= 9):
            raise ValueError(f"barrier digit must be 0-9, got {self.barrier}")
        if self.duration_ticks < 1 or self.duration_ticks > 10:
            raise ValueError(f"duration_ticks must be between 1 and 10, got {self.duration_ticks}")
