"""
Project APEX
Configuration Manager
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Loads configuration from config.yaml."""

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config_path = root / "project_apex" / "config" / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)

    def get(self, *keys: str) -> Any:
        value: Any = self._config

        for key in keys:
            value = value[key]

        return value