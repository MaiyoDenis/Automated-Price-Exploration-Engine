"""
Project APEX
Configuration Manager

Loads ``config.yaml`` at startup and exposes both a generic accessor and
typed accessors for the most common Python types.  A missing or ``None``
value from a typed accessor raises :class:`~project_apex.exceptions.ConfigurationError`
immediately, so configuration problems surface at startup rather than at
the first use of the value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from project_apex.exceptions import ConfigurationError


class Config:
    """Loads configuration from ``config.yaml`` and provides typed accessors.

    Usage::

        config = Config()

        # Generic (backward-compatible)
        db_path = config.get("database", "path")

        # Typed accessors — raise ConfigurationError on missing / None
        url = config.get_str("api", "websocket_url")
        interval = config.get_int("api", "heartbeat_interval")
        delay = config.get_float("api", "reconnect_initial_delay")
        symbols = config.get_list("market", "symbols")
    """

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config_path = root / "project_apex" / "config" / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as file:
            self._config: dict[str, Any] = yaml.safe_load(file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, *keys: str) -> Any:
        """Traverse the nested config dict and return the value at *keys*.

        Args:
            *keys: Sequence of string keys forming the path, e.g.
                ``("api", "heartbeat_interval")``.

        Returns:
            The raw value stored at the given path.

        Raises:
            ConfigurationError: If any key in the path is not found or if
                the final value is ``None``.
        """
        key_path = ".".join(keys)
        value: Any = self._config

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ConfigurationError(key_path)
            value = value[key]

        if value is None:
            raise ConfigurationError(key_path)

        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, *keys: str) -> Any:
        """Return the raw value at the given key path (backward-compatible).

        This method preserves the original behaviour: it raises a plain
        ``KeyError`` (via dict access) if the key is absent.  Use the typed
        accessors for new code.

        Args:
            *keys: Sequence of string keys forming the path.

        Returns:
            The value stored at the given path (any type).
        """
        value: Any = self._config

        for key in keys:
            value = value[key]

        return value

    def get_str(self, *keys: str) -> str:
        """Return the configuration value at *keys* as a ``str``.

        Args:
            *keys: Sequence of string keys forming the path.

        Returns:
            The value cast to ``str``.

        Raises:
            ConfigurationError: If the key path is missing or the value is
                ``None``.
        """
        return str(self._resolve(*keys))

    def get_int(self, *keys: str) -> int:
        """Return the configuration value at *keys* as an ``int``.

        Args:
            *keys: Sequence of string keys forming the path.

        Returns:
            The value cast to ``int``.

        Raises:
            ConfigurationError: If the key path is missing or the value is
                ``None``.
            ValueError: If the stored value cannot be converted to ``int``.
        """
        return int(self._resolve(*keys))

    def get_float(self, *keys: str) -> float:
        """Return the configuration value at *keys* as a ``float``.

        Args:
            *keys: Sequence of string keys forming the path.

        Returns:
            The value cast to ``float``.

        Raises:
            ConfigurationError: If the key path is missing or the value is
                ``None``.
            ValueError: If the stored value cannot be converted to ``float``.
        """
        return float(self._resolve(*keys))

    def get_list(self, *keys: str) -> list:
        """Return the configuration value at *keys* as a ``list``.

        Args:
            *keys: Sequence of string keys forming the path.

        Returns:
            The value as a ``list``.

        Raises:
            ConfigurationError: If the key path is missing, the value is
                ``None``, or the value is not a list.
        """
        key_path = ".".join(keys)
        value = self._resolve(*keys)

        if not isinstance(value, list):
            raise ConfigurationError(key_path)

        return value
