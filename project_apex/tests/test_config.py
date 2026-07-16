"""
Test Config
"""

import pytest

from project_apex.config.config import Config
from project_apex.exceptions import ConfigurationError


@pytest.fixture
def config():
    return Config()


def test_get_str_returns_string(config: Config):
    assert isinstance(config.get_str("application", "name"), str)


def test_get_int_returns_int(config: Config):
    assert isinstance(config.get_int("api", "heartbeat_interval"), int)


def test_get_float_returns_float(config: Config):
    assert isinstance(config.get_float("api", "reconnect_initial_delay"), float)


def test_get_list_returns_list(config: Config):
    assert isinstance(config.get_list("market", "symbols"), list)


def test_missing_key_raises_configuration_error(config: Config):
    with pytest.raises(ConfigurationError):
        config.get_str("nonexistent", "key")
