"""
Project APEX
Deriv Client (MarketDataProvider implementation)
"""

from __future__ import annotations

import asyncio
import aiohttp
from typing import Optional

from loguru import logger

from project_apex.api.ports import MarketDataProvider
from project_apex.api.websocket_client import WebSocketManager, ConnectionState
from project_apex.api.messages import MessageBuilder, MessageParser
from project_apex.config.config import Config
from project_apex.exceptions import ConnectionFailedError, DerivAPIError
from project_apex.models.tick import Tick


class DerivClient(MarketDataProvider):
    """Concrete MarketDataProvider for the Deriv API."""

    def __init__(self, config: Config) -> None:
        self._config = config
        from project_apex.config.environment import Environment
        self._env = Environment()
        
        url = self._config.get_str("api", "websocket_url")
        app_id = self._env.app_id
        self._full_url = f"{url}?app_id={app_id}"
        
        self._ws = WebSocketManager(
            url=self._full_url,
            connect_timeout=self._config.get_float("api", "heartbeat_timeout")
        )
        self._builder = MessageBuilder()
        self._parser = MessageParser()
        
        self._queue: asyncio.Queue[Tick] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        self._max_attempts = self._config.get_int("api", "reconnect_max_attempts")
        self._initial_delay = self._config.get_float("api", "reconnect_initial_delay")
        self._max_delay = self._config.get_float("api", "reconnect_max_delay")
        
        try:
            self._heartbeat_interval = self._config.get_float("api", "heartbeat_interval")
        except Exception:
            self._heartbeat_interval = 30.0

        self._active_subscriptions: set[str] = set()

    async def _fetch_otp_url(self) -> str:
        """Fetch the temporary WebSocket URL using the OTP REST endpoint."""
        rest_url = self._config.get_str("api", "rest_url")
        is_paper_trading = self._config.get("trading", "paper_trading")
        
        if is_paper_trading:
            account_id = self._env.demo_account_id
            token = self._env.demo_token
            logger.info("Connecting with DEMO account credentials.")
        else:
            account_id = self._env.real_account_id
            token = self._env.real_token
            logger.info("Connecting with REAL account credentials.")
            
        app_id = self._env.app_id
        
        url = f"{rest_url}/trading/v1/options/accounts/{account_id}/otp"
        headers = {
            "Authorization": f"Bearer {token}",
            "Deriv-App-ID": app_id
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ConnectionFailedError(f"OTP Handshake failed ({response.status}): {text}")
                data = await response.json()
                return data["data"]["url"]

    @property
    def is_connected(self) -> bool:
        return self._ws.state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        dynamic_url = await self._fetch_otp_url()
        self._ws.url = dynamic_url
        await self._ws.connect()
        self._start_receive_loop()

    async def disconnect(self) -> None:
        if getattr(self, "_heartbeat_task", None) is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        await self._ws.disconnect()

    async def subscribe_ticks(self, symbol: str) -> str:
        req = self._builder.subscribe_ticks(symbol)
        await self._ws.send(req)
        self._active_subscriptions.add(symbol)
        return symbol  # Returning symbol as subscription id for simplicity since API uses it

    async def unsubscribe(self, subscription_id: str) -> None:
        # Deriv forget uses the specific stream id, but we can also use forget_all for ticks.
        # For this prototype, we'll just send forget request.
        req = self._builder.unsubscribe(subscription_id)
        await self._ws.send(req)
        if subscription_id in self._active_subscriptions:
            self._active_subscriptions.remove(subscription_id)

    async def receive(self) -> Tick:
        return await self._queue.get()

    def _start_receive_loop(self) -> None:
        if self._receive_task is None or self._receive_task.done():
            self._receive_task = asyncio.create_task(self._receive_loop())
            
        if getattr(self, "_heartbeat_task", None) is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self.is_connected:
                    req = self._builder.ping()
                    await self._ws.send(req)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")

    async def _receive_loop(self) -> None:
        while True:
            try:
                raw_msg = await self._ws.receive()
                parsed = self._parser.parse(raw_msg)
                
                if isinstance(parsed, Tick):
                    await self._queue.put(parsed)
                elif isinstance(parsed, list):
                    # For history ticks/candles, we might need to handle differently, 
                    # but MarketDataProvider interface specifies `receive() -> Tick`.
                    for item in parsed:
                        if isinstance(item, Tick):
                            await self._queue.put(item)
                            
            except DerivAPIError as e:
                logger.error(f"Deriv API Error: {e.code} - {e.message}")
            except Exception as e:
                logger.warning(f"WebSocket receive error: {e}. Initiating reconnect.")
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())
                break # Exit current loop to let reconnect handle it

    async def _reconnect_loop(self) -> None:
        delay = self._initial_delay
        attempts = 0
        
        while attempts < self._max_attempts:
            try:
                logger.info(f"Reconnecting attempt {attempts + 1}/{self._max_attempts} after {delay}s...")
                await asyncio.sleep(delay)
                
                dynamic_url = await self._fetch_otp_url()
                self._ws.url = dynamic_url
                await self._ws.connect()
                
                # Resubscribe active subscriptions
                for symbol in self._active_subscriptions:
                    req = self._builder.subscribe_ticks(symbol)
                    await self._ws.send(req)
                
                logger.success("Reconnected successfully.")
                self._start_receive_loop()
                return
                
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
                attempts += 1
                delay = min(delay * 2, self._max_delay)
                
        logger.critical("Max reconnection attempts exhausted.")
        raise ConnectionFailedError("Max reconnection attempts exhausted.")
