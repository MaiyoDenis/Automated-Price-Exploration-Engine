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

        # Maps symbol → Deriv subscription UUID (from subscription.id in tick response).
        # Required for the `forget` request to actually unsubscribe the stream.
        self._subscription_ids: dict[str, str] = {}   # symbol → uuid
        self._active_subscriptions: set[str] = set()  # legacy: symbols subscribed

        # Pending RPC requests: req_id → Future
        self._pending_requests: dict[int, asyncio.Future] = {}

        # Contract update push handlers: contract_id (str) → asyncio.Queue
        # DiffersExecutor registers a queue here; pushed proposal_open_contract
        # messages are placed on the queue for non-blocking settlement processing.
        self._contract_update_queues: dict[str, asyncio.Queue] = {}

    # ── Auth & Connect ────────────────────────────────────────────────────────

    async def _fetch_otp_url(self) -> str:
        """Fetch the temporary WebSocket URL using the OTP REST endpoint."""
        rest_url = self._config.get_str("api", "rest_url")

        try:
            account_type = self._config.get("api", "account_type")
        except Exception:
            account_type = "demo"

        if account_type == "demo":
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

    async def _authorize(self) -> None:
        """
        Send the authorize message and wait for its response.

        This MUST be called after every (re)connect before any trading calls.
        Without it, proposal/buy/sell return AuthorizationRequired errors.
        """
        try:
            account_type = self._config.get("api", "account_type") or "demo"
        except Exception:
            account_type = "demo"

        token = self._env.demo_token if account_type == "demo" else self._env.real_token

        req = self._builder.authorize(token)
        response = await self.send_request(req)
        account_info = response.get("authorize", {})
        logger.success(
            f"[DerivClient] Authorized | account={account_info.get('loginid', '?')} "
            f"balance={account_info.get('balance', '?')} {account_info.get('currency', '')}"
        )

    @property
    def is_connected(self) -> bool:
        return self._ws.state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """Connect the WebSocket and authorize before returning."""
        dynamic_url = await self._fetch_otp_url()
        self._ws.url = dynamic_url
        await self._ws.connect()
        self._start_receive_loop()
        # Authorize immediately — every trading call requires this
        await self._authorize()

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

    # ── Subscriptions ─────────────────────────────────────────────────────────

    async def subscribe_ticks(self, symbol: str) -> str:
        """
        Subscribe to tick stream for a symbol.

        Returns the symbol string as the public subscription handle, but
        internally stores the Deriv subscription UUID so forget() works.
        The UUID is extracted from the first tick response for this symbol.
        """
        req = self._builder.subscribe_ticks(symbol)
        await self._ws.send(req)
        self._active_subscriptions.add(symbol)
        return symbol  # caller uses symbol as handle; UUID stored in _subscription_ids

    async def unsubscribe(self, symbol: str) -> None:
        """
        Send a `forget` request using the real Deriv subscription UUID.

        Falls back to symbol-based forget if we haven't received a tick yet
        and don't have the UUID (best-effort).
        """
        uuid = self._subscription_ids.get(symbol)
        if uuid:
            req = self._builder.unsubscribe(uuid)
            try:
                await self._ws.send(req)
                logger.info(f"[DerivClient] Unsubscribed {symbol} (uuid={uuid[:8]}...)")
            except Exception as exc:
                logger.warning(f"[DerivClient] Forget failed for {symbol}: {exc}")
            self._subscription_ids.pop(symbol, None)
        else:
            # No UUID yet — symbol may not have sent a tick; try symbol as fallback
            logger.warning(
                f"[DerivClient] No subscription UUID for {symbol} — "
                f"sending symbol-based forget (best-effort)"
            )
            req = self._builder.unsubscribe(symbol)
            try:
                await self._ws.send(req)
            except Exception as exc:
                logger.warning(f"[DerivClient] Fallback forget failed for {symbol}: {exc}")

        self._active_subscriptions.discard(symbol)

    async def receive(self) -> Tick:
        return await self._queue.get()

    def register_contract_update_handler(
        self, contract_id: str, queue: "asyncio.Queue[dict]"
    ) -> None:
        """
        Register a queue to receive pushed proposal_open_contract updates.

        Called by DiffersExecutor after buying a contract so that server-pushed
        settlement messages are forwarded to the executor's settlement listener
        without polling.
        """
        self._contract_update_queues[str(contract_id)] = queue
        logger.debug(f"[DerivClient] Registered contract update handler for {contract_id}")

    def unregister_contract_update_handler(self, contract_id: str) -> None:
        """Remove the push handler for a settled contract."""
        self._contract_update_queues.pop(str(contract_id), None)
        logger.debug(f"[DerivClient] Unregistered contract update handler for {contract_id}")

    # ── RPC ───────────────────────────────────────────────────────────────────

    async def send_request(self, request_msg: dict) -> dict:
        """Sends an RPC request over the WebSocket and awaits the correlated response."""
        req_id = request_msg.get("req_id")
        if req_id is None:
            raise ValueError("Request message must contain a req_id to await a response.")

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[req_id] = future

        try:
            await self._ws.send(request_msg)
            response = await asyncio.wait_for(future, timeout=10.0)

            if "error" in response:
                error = response["error"]
                raise DerivAPIError(error.get("code", "UNKNOWN"), error.get("message", "Unknown API Error"))

            return response
        finally:
            self._pending_requests.pop(req_id, None)

    # ── Internal loops ────────────────────────────────────────────────────────

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

                # Route pending RPC responses by req_id before domain parsing
                if "req_id" in raw_msg:
                    req_id = raw_msg["req_id"]
                    if req_id in self._pending_requests:
                        if not self._pending_requests[req_id].done():
                            self._pending_requests[req_id].set_result(raw_msg)
                        continue

                # Extract and store the Deriv subscription UUID from tick messages.
                # This UUID is required for the forget (unsubscribe) request.
                if raw_msg.get("msg_type") == "tick":
                    sub = raw_msg.get("subscription", {})
                    uuid = sub.get("id")
                    symbol = raw_msg.get("tick", {}).get("symbol")
                    if uuid and symbol and symbol not in self._subscription_ids:
                        self._subscription_ids[symbol] = uuid
                        logger.debug(
                            f"[DerivClient] Stored subscription UUID for {symbol}: {uuid[:8]}..."
                        )

                # Route pushed proposal_open_contract updates to registered queues.
                # These are server-pushed settlement messages (not correlated by req_id).
                if raw_msg.get("msg_type") == "proposal_open_contract":
                    contract_info = raw_msg.get("proposal_open_contract", {})
                    contract_id = str(contract_info.get("contract_id", ""))
                    queue = self._contract_update_queues.get(contract_id)
                    if queue is not None:
                        await queue.put(contract_info)
                        continue  # Don't pass to domain parser

                parsed = self._parser.parse(raw_msg)

                if isinstance(parsed, Tick):
                    await self._queue.put(parsed)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, Tick):
                            await self._queue.put(item)

            except DerivAPIError as e:
                logger.error(f"Deriv API Error: {e.code} - {e.message}")
            except Exception as e:
                logger.warning(f"WebSocket receive error: {e}. Initiating reconnect.")
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())
                break

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

                # Re-authorize before resubscribing
                await self._authorize()

                # Resubscribe active symbols; clear stale UUIDs so they get
                # repopulated from the next tick response.
                symbols_to_resub = list(self._active_subscriptions)
                self._subscription_ids.clear()
                for symbol in symbols_to_resub:
                    req = self._builder.subscribe_ticks(symbol)
                    await self._ws.send(req)

                logger.success("Reconnected and reauthorized successfully.")
                self._start_receive_loop()
                return

            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
                attempts += 1
                delay = min(delay * 2, self._max_delay)

        logger.critical("Max reconnection attempts exhausted.")
        raise ConnectionFailedError("Max reconnection attempts exhausted.")

    # ── Account helpers ───────────────────────────────────────────────────────

    async def get_account_balance(self, account_type: str = "demo") -> dict:
        """Fetch account balance. Uses the live connection if account_type matches."""
        try:
            connected_type = self._config.get("api", "account_type") or "demo"
        except Exception:
            connected_type = "demo"

        if account_type == connected_type and self.is_connected:
            try:
                req = self._builder.balance()
                response = await self.send_request(req)
                balance_data = response.get("balance", {})
                return {
                    "balance": float(balance_data.get("balance", 0.0)),
                    "currency": balance_data.get("currency", "USD"),
                    "account_type": account_type,
                }
            except Exception as e:
                logger.warning(f"[DerivClient] Balance fetch error: {e}")
                return {"balance": None, "currency": "USD", "account_type": account_type}
        else:
            return await self._fetch_balance_via_temp_connection(account_type)

    async def _fetch_balance_via_temp_connection(self, account_type: str) -> dict:
        """Open a temporary WS session to fetch balance for the alternate account type."""
        rest_url = self._config.get_str("api", "rest_url")
        app_id = self._env.app_id

        if account_type == "demo":
            account_id = self._env.demo_account_id
            token = self._env.demo_token
        else:
            account_id = self._env.real_account_id
            token = self._env.real_token

        try:
            url = f"{rest_url}/trading/v1/options/accounts/{account_id}/otp"
            headers = {"Authorization": f"Bearer {token}", "Deriv-App-ID": app_id}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"[DerivClient] OTP failed for {account_type}: {text}")
                        return {"balance": None, "currency": "USD", "account_type": account_type}
                    data = await resp.json()
                    ws_url = data["data"]["url"]

            import json
            import websockets
            async with websockets.connect(ws_url) as ws:
                builder = MessageBuilder()
                # Authorize first on this temp connection
                auth_req = builder.authorize(token)
                await ws.send(json.dumps(auth_req))
                # Then request balance
                bal_req = builder.balance()
                await ws.send(json.dumps(bal_req))

                for _ in range(10):
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg = json.loads(raw)
                    if msg.get("msg_type") == "balance":
                        bal = msg.get("balance", {})
                        return {
                            "balance": float(bal.get("balance", 0.0)),
                            "currency": bal.get("currency", "USD"),
                            "account_type": account_type,
                        }
        except Exception as e:
            logger.warning(f"[DerivClient] Temp balance fetch error ({account_type}): {e}")

        return {"balance": None, "currency": "USD", "account_type": account_type}
