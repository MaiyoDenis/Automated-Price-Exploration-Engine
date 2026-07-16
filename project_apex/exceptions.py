"""
Project APEX
Project-wide Exception Hierarchy

This module defines the base exception and all project-specific exceptions.
"""

from __future__ import annotations


class ApexError(Exception):
    """Base class for all Project APEX exceptions.

    All project-specific exceptions inherit from this class so that callers
    can catch the entire APEX exception family with a single ``except ApexError``
    clause when broad error handling is required.
    """


class ConfigurationError(ApexError):
    """Raised when a required configuration key is absent or its value is ``None``.

    Attributes:
        key_path: Dot-joined string identifying the missing key, e.g.
            ``"api.websocket_url"``.
    """

    def __init__(self, key_path: str) -> None:
        self.key_path = key_path
        super().__init__(f"Required configuration key is missing or None: '{key_path}'")


class AuthenticationError(ApexError):
    """Raised when Deriv `authorize` response contains an error."""


class DerivAPIError(ApexError):
    """Raised when any Deriv response contains a top-level `"error"` key.

    Attributes:
        code: Error code returned by Deriv.
        message: Error message returned by Deriv.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"Deriv API Error [{code}]: {message}")


class ConnectionFailedError(ApexError):
    """Raised when reconnection attempts are exhausted."""

