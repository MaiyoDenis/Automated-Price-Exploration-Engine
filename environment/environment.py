"""
Project APEX

Environment Manager
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


class Environment:
    """Loads environment variables."""

    def __init__(self) -> None:
        load_dotenv()

    @property
    def app_id(self) -> str:
        value = os.getenv("DERIV_APP_ID")

        if not value:
            raise ValueError("DERIV_APP_ID is missing.")

        return value

    @property
    def demo_token(self) -> str:
        value = os.getenv("DERIV_DEMO_TOKEN")

        if not value:
            raise ValueError("DERIV_DEMO_TOKEN is missing.")

        return value

    @property
    def real_token(self) -> str:
        value = os.getenv("DERIV_REAL_TOKEN")

        if not value:
            raise ValueError("DERIV_REAL_TOKEN is missing.")

        return value