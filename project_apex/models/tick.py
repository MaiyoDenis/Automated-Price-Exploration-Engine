"""
Project APEX
Domain Models: Tick

This module defines the immutable Tick domain object.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tick:
    """A single price event for one instrument.

    Attributes:
        symbol: The instrument identifier (e.g., "R_25").
        timestamp: Unix epoch timestamp in milliseconds.
        price: The current price quote.
    """

    symbol: str
    timestamp: int
    price: float
