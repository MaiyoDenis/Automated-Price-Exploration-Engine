"""
Project APEX
Environment Manager

Loads environment variables from the .env file and validates them eagerly
at construction time so that missing credentials surface immediately at
startup rather than deep inside connect() calls.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger


_REQUIRED_ALWAYS = [
    "DERIV_APP_ID",
    "DERIV_DEMO_ACCOUNT_ID",
    "DERIV_DEMO_TOKEN",
]

_REQUIRED_LIVE = [
    "DERIV_REAL_ACCOUNT_ID",
    "DERIV_REAL_TOKEN",
]


class Environment:
    """
    Environment variable manager.

    Validates credentials at construction time so that missing values
    cause a clear startup error rather than a cryptic failure deep inside
    a network call.

    Args:
        require_live_credentials: When True (set in non-paper-trading mode),
            also validates DERIV_REAL_ACCOUNT_ID and DERIV_REAL_TOKEN.
    """

    def __init__(self, require_live_credentials: bool = False) -> None:
        load_dotenv()
        self._validate(require_live_credentials)

    def _validate(self, require_live: bool) -> None:
        """Check all required env vars are present and non-empty. Raise on first failure."""
        missing: list[str] = []

        for key in _REQUIRED_ALWAYS:
            if not os.getenv(key):
                missing.append(key)

        if require_live:
            for key in _REQUIRED_LIVE:
                if not os.getenv(key):
                    missing.append(key)

        if missing:
            raise EnvironmentError(
                f"[Environment] Missing required environment variables: {missing}. "
                f"Check your .env file."
            )

        logger.info(
            f"[Environment] Credentials validated | "
            f"live_required={require_live}"
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def app_id(self) -> str:
        value = os.getenv("DERIV_APP_ID")
        if not value:
            raise EnvironmentError("DERIV_APP_ID is not set.")
        return value

    @property
    def demo_account_id(self) -> str:
        value = os.getenv("DERIV_DEMO_ACCOUNT_ID")
        if not value:
            raise EnvironmentError("DERIV_DEMO_ACCOUNT_ID is not set.")
        return value

    @property
    def real_account_id(self) -> str:
        value = os.getenv("DERIV_REAL_ACCOUNT_ID")
        if not value:
            raise EnvironmentError("DERIV_REAL_ACCOUNT_ID is not set.")
        return value

    @property
    def demo_token(self) -> str:
        value = os.getenv("DERIV_DEMO_TOKEN")
        if not value:
            raise EnvironmentError("DERIV_DEMO_TOKEN is not set.")
        return value

    @property
    def real_token(self) -> str:
        value = os.getenv("DERIV_REAL_TOKEN")
        if not value:
            raise EnvironmentError("DERIV_REAL_TOKEN is not set.")
        return value

    @property
    def environment(self) -> str:
        return os.getenv("ENVIRONMENT", "development")
