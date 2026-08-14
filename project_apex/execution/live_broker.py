"""
Project APEX — Live Broker

Real order execution via the Deriv API.
"""
from __future__ import annotations

import time
from loguru import logger

from project_apex.risk.models import TradeOrder, Direction
from project_apex.execution.models import Position, Trade
from project_apex.execution.paper_broker import PaperBroker
from project_apex.api.deriv_client import DerivClient
from project_apex.api.messages import MessageBuilder


class LiveBroker(PaperBroker):
    """
    Live order execution broker (Deriv API).
    """

    def __init__(
        self,
        client: DerivClient,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0
    ) -> None:
        super().__init__(slippage_pct=slippage_pct, spread_pct=spread_pct)
        self.client = client
        self.builder = MessageBuilder()
        logger.info("[LiveBroker] Initialized in LIVE mode connected to DerivClient.")

    async def open_position(self, order: TradeOrder, current_time_ms: int | None = None) -> Position:
        """Execute a live buy order."""
        contract_type = "CALL" if order.direction == Direction.LONG else "PUT"
        
        # 1. Proposal
        try:
            prop_req = self.builder.proposal(order.symbol, order.size, contract_type)
            prop_resp = await self.client.send_request(prop_req)
            proposal_id = prop_resp["proposal"]["id"]
        except Exception as e:
            logger.error(f"[LiveBroker] Proposal failed: {e}")
            raise
            
        # 2. Buy
        try:
            buy_req = self.builder.buy(proposal_id, order.size)
            buy_resp = await self.client.send_request(buy_req)
            
            contract_id = str(buy_resp["buy"]["contract_id"])
            buy_price = float(buy_resp["buy"]["buy_price"])
        except Exception as e:
            logger.error(f"[LiveBroker] Buy failed: {e}")
            raise

        logger.info(
            f"[LiveBroker] LIVE OPENED {order.direction.name} {order.symbol} | "
            f"contract_id={contract_id} price={buy_price}"
        )
        
        pos = Position(
            id=contract_id,
            symbol=order.symbol,
            direction=order.direction,
            size=order.size,
            entry_price=buy_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            strategy_name=order.strategy_name,
            opened_at=current_time_ms or int(time.time() * 1000)
        )
        self.open_positions[pos.id] = pos
        return pos

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
        current_time_ms: int | None = None,
    ) -> Trade | None:
        """Execute a live sell order."""
        if position_id not in self.open_positions:
            return None
            
        pos = self.open_positions[position_id]
        
        try:
            sell_req = self.builder.sell(int(position_id))
            sell_resp = await self.client.send_request(sell_req)
            
            sold_for = float(sell_resp["sell"]["sold_for"])
        except Exception as e:
            logger.error(f"[LiveBroker] Sell failed: {e}")
            raise
            
        self.open_positions.pop(position_id)
        
        realized_pnl = sold_for - pos.entry_price
        realized_pnl_pct = realized_pnl / pos.entry_price if pos.entry_price else 0.0

        trade = Trade(
            id=pos.id,
            symbol=pos.symbol,
            direction=pos.direction,
            size=pos.size,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            opened_at=pos.opened_at,
            closed_at=current_time_ms or int(time.time() * 1000),
            close_reason=reason,
            strategy_name=pos.strategy_name,
        )
        self.trade_history.append(trade)
        
        logger.info(
            f"[LiveBroker] LIVE CLOSED {pos.direction.name} {pos.symbol} | "
            f"contract_id={position_id} sold_for={sold_for} PnL={realized_pnl:.2f} reason={reason}"
        )
        
        await self._fire_trade_callbacks(trade)
        return trade
