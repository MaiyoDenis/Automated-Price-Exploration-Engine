"""
Project APEX
WebSocket Client Test
"""

from __future__ import annotations

import asyncio

from project_apex.api.websocket_client import WebSocketClient
from project_apex.config.environment import Environment


async def main() -> None:
    """Test the WebSocket connection."""

    env = Environment()

    url = f"wss://ws.derivws.com/websockets/v3?app_id={env.app_id}"

    client = WebSocketClient(url)

    await client.connect()

    print("Connected successfully!")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())