"""
Project APEX
Generic WebSocket Manager
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum, auto

import websockets
from loguru import logger


class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()


class WebSocketManager:
    """Generic asynchronous WebSocket client and context manager."""

    def __init__(self, url: str, connect_timeout: float = 30.0) -> None:
        self.url = url
        self.connect_timeout = connect_timeout
        self.connection = None
        self._state = ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        """Returns the current connection state."""
        return self._state

    async def connect(self) -> None:
        """Connect to the WebSocket server with a timeout."""
        if self._state == ConnectionState.CONNECTED:
            return

        self._state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.url} (timeout: {self.connect_timeout}s)")

        try:
            async with asyncio.timeout(self.connect_timeout):
                self.connection = await websockets.connect(self.url)
            self._state = ConnectionState.CONNECTED
            logger.success("WebSocket connected successfully.")
        except Exception:
            self._state = ConnectionState.DISCONNECTED
            raise

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

        self._state = ConnectionState.DISCONNECTED
        logger.success("WebSocket disconnected.")

    async def send(self, message: dict) -> None:
        """Send a JSON message."""
        if self.connection is None or self._state != ConnectionState.CONNECTED:
            raise RuntimeError("WebSocket is not connected.")

        await self.connection.send(json.dumps(message))

    async def receive(self) -> dict:
        """Receive a JSON message."""
        if self.connection is None or self._state != ConnectionState.CONNECTED:
            raise RuntimeError("WebSocket is not connected.")

        response = await self.connection.recv()
        return json.loads(response)

    async def __aenter__(self) -> WebSocketManager:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()