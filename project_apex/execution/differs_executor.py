"""
Project APEX — Differs Executor

Executes DIGITDIFF contracts and tracks settlement via a push subscription
(proposal_open_contract with subscribe:1) rather than polling.

Push subscription means:
  - The Deriv server sends an update the moment the contract settles.
  - No polling delay — 1-tick contracts (~1s) are caught immediately.
  - No risk of duplicate settlement on reconnect (server pushes once).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional, Callable
from loguru import logger

from project_apex.api.deriv_client import DerivClient
from project_apex.api.messages import MessageBuilder
from project_apex.risk.models import TradeOrder
from project_apex.execution.portfolio import Portfolio
from project_apex.execution.models import Trade


# Sentinel placed in the queue when the executor is stopping
_STOP_SENTINEL = object()


class DiffersExecutor:
    """
    Executes DIGITDIFF contracts via the Deriv API.

    Flow:
      1. digit_proposal → get proposal_id
      2. buy → get contract_id
      3. Subscribe to proposal_open_contract with subscribe:1 so the server
         pushes the settlement result instead of us polling for it.
      4. On settlement push, call the settlement callback and update portfolio.
    """

    def __init__(self, provider: DerivClient, portfolio: Portfolio) -> None:
        self.provider = provider
        self.portfolio = portfolio
        self.builder = MessageBuilder()

        # Currently active contract — only one DIGITDIFF at a time
        self.active_contract_id: Optional[str] = None
        self.active_order: Optional[TradeOrder] = None
        self._is_executing: bool = False

        # Settlement callback: (won: bool, order: TradeOrder | None) -> None
        self.on_contract_settled: Optional[Callable[[bool, Optional[TradeOrder]], None]] = None

        # Queue that the contract-update push stream writes into
        self._update_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._monitor_task: Optional[asyncio.Task] = None

    def set_settlement_callback(self, cb: Callable[[bool, Optional[TradeOrder]], None]) -> None:
        self.on_contract_settled = cb

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._settlement_listener())
        logger.info("[DiffersExecutor] Started push-based settlement listener.")

    async def stop(self) -> None:
        if self._monitor_task:
            # Unblock the listener queue so the task exits cleanly
            await self._update_queue.put(_STOP_SENTINEL)
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("[DiffersExecutor] Stopped.")

    # ── Order execution ───────────────────────────────────────────────────────

    async def execute_order(self, order: TradeOrder) -> None:
        """
        Execute a DIGITDIFF order: proposal → buy → subscribe to settlement.

        Only one contract is active at a time. Concurrent calls are rejected
        until the current contract settles.
        """
        if self.active_contract_id is not None or self._is_executing:
            logger.warning("[DiffersExecutor] Rejecting order: already have an active contract.")
            return

        self._is_executing = True

        barrier = order.metadata.get("barrier")
        duration = order.metadata.get("duration_ticks", 1)
        stake = order.size

        if barrier is None:
            logger.error("[DiffersExecutor] TradeOrder missing 'barrier' metadata. Aborting.")
            self._is_executing = False
            return

        logger.info(
            f"[DiffersExecutor] Executing DIGITDIFF | "
            f"symbol={order.symbol} digit!={barrier} stake=${stake}"
        )

        try:
            # 1. Proposal
            prop_req = self.builder.digit_proposal(
                symbol=order.symbol,
                amount=stake,
                contract_type="DIGITDIFF",
                barrier=barrier,
                duration_ticks=duration,
            )
            prop_resp = await self.provider.send_request(prop_req)
            proposal = prop_resp.get("proposal")
            if not proposal:
                logger.error(f"[DiffersExecutor] No proposal in response: {prop_resp}")
                self._is_executing = False
                return

            proposal_id = proposal["id"]

            # 2. Buy
            buy_req = self.builder.buy(proposal_id=proposal_id, price=stake)
            buy_resp = await self.provider.send_request(buy_req)
            buy_result = buy_resp.get("buy")
            if not buy_result:
                logger.error(f"[DiffersExecutor] Buy failed: {buy_resp}")
                self._is_executing = False
                return

            contract_id = str(buy_result["contract_id"])
            self.active_contract_id = contract_id
            self.active_order = order

            logger.info(
                f"[DiffersExecutor] DIGITDIFF bought | contract_id={contract_id}"
            )

            # 3. Subscribe to settlement push (subscribe:1 means server pushes updates)
            asyncio.create_task(self._subscribe_contract_updates(contract_id))

        except Exception as exc:
            logger.error(f"[DiffersExecutor] Execution failed: {exc}", exc_info=True)
        finally:
            self._is_executing = False

    async def _subscribe_contract_updates(self, contract_id: str) -> None:
        """
        Send a proposal_open_contract request with subscribe:1.

        The Deriv server will push a message every time the contract state
        changes, and a final message when it settles (is_sold=1).
        Each pushed message is routed here via DerivClient's receive loop
        through the pending_requests correlation map, then placed on
        _update_queue for the settlement listener to process.
        """
        req = {
            "proposal_open_contract": 1,
            "contract_id": int(contract_id),
            "subscribe": 1,
            "req_id": self.builder._next_req_id(),
        }
        try:
            # send_request awaits the FIRST response (the initial snapshot).
            # Subsequent push updates arrive via the receive loop and are placed
            # on _update_queue by the _route_contract_update() call registered below.
            initial = await self.provider.send_request(req)
            contract_info = initial.get("proposal_open_contract", {})
            # Check if it already settled in the initial response
            if contract_info.get("is_sold"):
                await self._update_queue.put(contract_info)
            else:
                # Register for push updates by hooking into the receive loop
                self.provider.register_contract_update_handler(
                    contract_id, self._update_queue
                )
        except Exception as exc:
            logger.error(
                f"[DiffersExecutor] Contract subscription failed "
                f"(contract_id={contract_id}): {exc}"
            )

    # ── Settlement listener ───────────────────────────────────────────────────

    async def _settlement_listener(self) -> None:
        """
        Drain _update_queue and process settlement messages.

        Blocks waiting for items the push subscription places on the queue.
        Exits cleanly when the stop sentinel is received.
        """
        while True:
            try:
                item = await self._update_queue.get()

                if item is _STOP_SENTINEL:
                    break

                contract_info = item
                is_sold = contract_info.get("is_sold", 0)
                if not is_sold:
                    continue  # Intermediate update (price move etc.) — ignore

                status = contract_info.get("status", "")
                profit = float(contract_info.get("profit", 0.0))
                won = status == "won"

                logger.info(
                    f"[DiffersExecutor] Contract {self.active_contract_id} settled | "
                    f"result={status.upper()} PnL=${profit:+.2f}"
                )

                # Capture and clear active state FIRST to prevent race conditions
                settled_contract_id = self.active_contract_id
                settled_order = self.active_order
                self.active_contract_id = None
                self.active_order = None

                # Unregister from push updates for this contract
                if settled_contract_id:
                    self.provider.unregister_contract_update_handler(settled_contract_id)

                # Record in portfolio
                if settled_order:
                    stake = settled_order.size
                    pnl_pct = profit / stake if stake else 0.0
                    trade = Trade(
                        id=str(uuid.uuid4()),
                        symbol=settled_order.symbol,
                        direction=settled_order.direction,
                        size=stake,
                        entry_price=1.0,
                        exit_price=1.0 + pnl_pct,
                        opened_at=int(time.time() * 1000),
                        closed_at=int(time.time() * 1000),
                        realized_pnl=profit,
                        realized_pnl_pct=pnl_pct,
                        close_reason="settled",
                        strategy_name=settled_order.strategy_name,
                    )
                    self.portfolio.on_trade_closed(trade)

                # Notify strategy and risk engine
                if self.on_contract_settled:
                    self.on_contract_settled(won, settled_order)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception(f"[DiffersExecutor] Settlement listener error: {exc}")
