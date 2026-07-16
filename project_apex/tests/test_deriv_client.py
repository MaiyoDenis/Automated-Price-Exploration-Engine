"""
Test Deriv Client
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from project_apex.api.deriv_client import DerivClient
from project_apex.config.config import Config
from project_apex.exceptions import AuthenticationError, ConnectionFailedError
from project_apex.api.websocket_client import ConnectionState


@pytest.fixture
def mock_config():
    # Provide a simple config override for testing
    config = Config()
    # Mocking internal config dict to avoid file dependencies in unit tests
    config._config = {
        "api": {
            "websocket_url": "wss://ws.binaryws.com/websockets/v3",
            "app_id": "1089",
            "heartbeat_interval": 30,
            "heartbeat_timeout": 10,
            "reconnect_max_attempts": 3,
            "reconnect_initial_delay": 0.1,
            "reconnect_max_delay": 1.0,
        }
    }
    return config


@pytest.fixture
def mock_ws_manager():
    with patch("project_apex.api.deriv_client.WebSocketManager") as mock:
        instance = mock.return_value
        instance.connect = AsyncMock()
        instance.disconnect = AsyncMock()
        instance.send = AsyncMock()
        instance.receive = AsyncMock()
        # Initial state disconnected
        instance.state = ConnectionState.DISCONNECTED
        yield instance


@pytest.fixture
def client(mock_config, mock_ws_manager):
    client = DerivClient(mock_config)
    client._ws = mock_ws_manager
    return client


async def _noop():
    """A coroutine that does nothing — used to suppress 'never awaited' warnings."""
    pass


@pytest.mark.asyncio
async def test_connect_sets_is_connected_true(client, mock_ws_manager):
    mock_ws_manager.state = ConnectionState.CONNECTED

    def _close_coro(coro):
        coro.close()  # prevents 'never awaited' warning
        f = asyncio.get_event_loop().create_future()
        f.cancel()
        return f

    with patch("asyncio.create_task", side_effect=_close_coro):
        await client.connect()
        assert client.is_connected is True
        mock_ws_manager.connect.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_sets_is_connected_false(client, mock_ws_manager):
    mock_ws_manager.state = ConnectionState.CONNECTED

    def _close_coro(coro):
        coro.close()
        f = asyncio.get_event_loop().create_future()
        f.cancel()
        return f

    with patch("asyncio.create_task", side_effect=_close_coro):
        await client.connect()

    mock_ws_manager.state = ConnectionState.DISCONNECTED
    await client.disconnect()

    assert client.is_connected is False
    mock_ws_manager.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_idempotent(client, mock_ws_manager):
    await client.disconnect()
    await client.disconnect()  # Should not raise


@pytest.mark.asyncio
async def test_reconnect_attempts_with_backoff(client, mock_ws_manager):
    mock_ws_manager.connect.side_effect = Exception("Connection error")
    
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(ConnectionFailedError):
            await client._reconnect_loop()
            
        # Should have tried max_attempts times (3)
        assert mock_ws_manager.connect.call_count == 3
        # Should have slept 3 times
        assert mock_sleep.call_count == 3
        # Verify backoff (initial 0.1)
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)
        mock_sleep.assert_any_call(0.4)


@pytest.mark.asyncio
async def test_reconnect_success_resets_counter(client, mock_ws_manager):
    """Reconnect should succeed on attempt 2 and stop retrying."""
    # Fails first time, succeeds second time; then receive loop would start but
    # we patch create_task to avoid the infinite receive loop hanging the test.
    mock_ws_manager.connect.side_effect = [Exception("Error"), None]
    mock_ws_manager.state = ConnectionState.CONNECTED

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("asyncio.create_task") as mock_create_task:
        await client._reconnect_loop()

        # connect called twice: once fail, once succeed
        assert mock_ws_manager.connect.call_count == 2
        # Sleep is called once per attempt (before each connect attempt)
        assert mock_sleep.call_count == 2
        # No ConnectionFailedError raised — test passes if we reach here
