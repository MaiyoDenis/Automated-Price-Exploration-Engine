"""
Project APEX
Environment Manager

Loads environment variables from the .env file.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


class Environment:
    """Environment variable manager."""

    def __init__(self) -> None:
        load_dotenv()

    @property
    def app_id(self) -> str:
        """Return the Deriv App ID."""

        value = os.getenv("DERIV_APP_ID")

        if value is None:
            raise ValueError("DERIV_APP_ID is not set.")

        return value

    @property
    def demo_token(self) -> str:
        """Return the Deriv demo API token."""

        value = os.getenv("DERIV_DEMO_TOKEN")

        if value is None:
            raise ValueError("DERIV_DEMO_TOKEN is not set.")

        return value

    @property
    def real_token(self) -> str:
        """Return the Deriv real API token."""

        value = os.getenv("DERIV_REAL_TOKEN")

        if value is None:
            raise ValueError("DERIV_REAL_TOKEN is not set.")

        return value

    @property
    def deriv_token(self) -> str:
        """Return the active Deriv API token (demo token for now).

        Returns the value of ``DERIV_DEMO_TOKEN`` from the ``.env`` file.
        Task 10 will wire in the logic to switch between demo and real tokens
        based on the configured trading mode.

        Raises:
            ValueError: If ``DERIV_DEMO_TOKEN`` is not set.
        """
        return self.demo_token

    @property
    def environment(self) -> str:
        """Return the application environment."""

        return os.getenv("ENVIRONMENT", "development")