"""
Project APEX — Differs Executor

Executes DIGITDIFF contracts.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional, Dict, Any
from loguru import logger

from project_apex.api.deriv_client import DerivClient
from project_apex.api.messages import MessageBuilder
from project_apex.risk.models import TradeOrder, Direction
from project_apex.execution.portfolio import Portfolio
from project_apex.execution.models import Trade

class DiffersExecutor:
    def __init__(self, provider: DerivClient, portfolio: Portfolio):
        self.provider = provider
        self.portfolio = portfolio
        self.builder = MessageBuilder()
        
        # Currently active contract if any
        self.active_contract_id: Optional[str] = None
        self.active_order: Optional[TradeOrder] = None
        self.on_contract_settled = None # Callback to notify risk engine: (won: bool) -> None
        
        # We need a long-running task to poll for open contracts or listen to proposal_open_contract stream
        self._monitor_task = None
        
    def set_settlement_callback(self, cb):
        self.on_contract_settled = cb
        
    async def start(self):
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[DiffersExecutor] Started contract monitor loop")

    async def stop(self):
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("[DiffersExecutor] Stopped")

    async def execute_order(self, order: TradeOrder) -> None:
        if self.active_contract_id is not None or getattr(self, '_is_executing', False):
            logger.warning("[DiffersExecutor] Rejecting order: already executing or have active contract.")
            return

        self._is_executing = True
        
        barrier = order.metadata.get("barrier")
        duration = order.metadata.get("duration_ticks", 1)
        stake = order.size
        
        if barrier is None:
            logger.error("[DiffersExecutor] TradeOrder missing 'barrier' metadata")
            self._is_executing = False
            return
            
        logger.info(f"[DiffersExecutor] Executing DIGITDIFF on {order.symbol} | digit != {barrier} | stake=${stake}")

        try:
            # 1. Get proposal
            req = self.builder.digit_proposal(
                symbol=order.symbol,
                amount=stake,
                contract_type="DIGITDIFF",
                barrier=barrier,
                duration_ticks=duration
            )
            resp = await self.provider.send_request(req)
            proposal = resp.get("proposal")
            if not proposal:
                logger.error(f"[DiffersExecutor] Failed to get proposal: {resp.get('error')}")
                self._is_executing = False
                return
                
            proposal_id = proposal.get("id")
            
            # 2. Buy
            buy_req = self.builder.buy(proposal_id=proposal_id, price=stake)
            buy_resp = await self.provider.send_request(buy_req)
            
            buy_result = buy_resp.get("buy")
            if not buy_result:
                logger.error(f"[DiffersExecutor] Buy failed: {buy_resp.get('error')}")
                self._is_executing = False
                return
                
            self.active_contract_id = buy_result.get("contract_id")
            self.active_order = order
            
            logger.info(f"[DiffersExecutor] DIGITDIFF bought successfully | Contract ID: {self.active_contract_id}")
            self._is_executing = False
            
        except Exception as e:
            logger.error(f"[DiffersExecutor] Execution failed: {e}")
            self._is_executing = False
            
    async def _monitor_loop(self):
        """Polls for the status of the active contract to see if it won or lost."""
        builder = MessageBuilder()
        while True:
            try:
                await asyncio.sleep(1.0)
                if not self.active_contract_id:
                    continue
                    
                req = {
                    "proposal_open_contract": 1,
                    "contract_id": self.active_contract_id,
                    "req_id": builder._next_req_id()
                }
                
                resp = await self.provider.send_request(req)
                contract_info = resp.get("proposal_open_contract")
                
                if not contract_info:
                    continue
                    
                is_sold = contract_info.get("is_sold")
                if is_sold:
                    status = contract_info.get("status")
                    profit = float(contract_info.get("profit", 0.0))
                    won = status == "won"
                    
                    logger.info(f"[DiffersExecutor] Contract {self.active_contract_id} settled | Result: {status.upper()} | PnL: ${profit:.2f}")
                    
                    # Capture and clear active state FIRST to prevent
                    # infinite retry if portfolio call fails.
                    settled_order = self.active_order
                    self.active_contract_id = None
                    self.active_order = None
                    
                    if settled_order:
                        stake = settled_order.size
                        pnl_pct = profit / stake if stake else 0.0
                        trade = Trade(
                            id=str(uuid.uuid4()),
                            symbol=settled_order.symbol,
                            direction=settled_order.direction,
                            size=stake,
                            entry_price=1.0,
                            exit_price=1.0 + profit / stake if stake else 1.0,
                            opened_at=int(time.time() * 1000),
                            closed_at=int(time.time() * 1000),
                            realized_pnl=profit,
                            realized_pnl_pct=pnl_pct,
                            close_reason="settled",
                            strategy_name=settled_order.strategy_name,
                        )
                        self.portfolio.on_trade_closed(trade)
                    
                    if self.on_contract_settled:
                        self.on_contract_settled(won, settled_order)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[DiffersExecutor] Monitor loop error: {e}")
