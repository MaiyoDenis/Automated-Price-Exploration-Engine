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

    Inherits PaperBroker for position tracking and stop/TP logic.
    Overrides open/close to use real Deriv API calls.

    Key differences from PaperBroker:
    - open_position: proposal → buy via DerivClient.send_request()
    - close_position: checks is_valid_to_sell before calling sell
    - check_stops_and_targets: called on every tick (not just candle close)
      via the tick_callback so stops fire at real-time tick prices
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
        """
        Execute a live buy order via Deriv: proposal → buy.

        Note on contract_type:
          - CALL / PUT are rise/fall contracts (binary options).
            They settle at expiry — SL/TP are managed manually via sell.
          - The `symbol` field (not `underlying_symbol`) is the correct
            Deriv API field for the proposal request.
        """
        contract_type = "CALL" if order.direction == Direction.LONG else "PUT"

        # 1. Proposal — use "symbol" not "underlying_symbol"
        try:
            prop_req = self.builder.proposal(order.symbol, order.size, contract_type)
            prop_resp = await self.client.send_request(prop_req)
            proposal_id = prop_resp["proposal"]["id"]
        except Exception as e:
            logger.error(f"[LiveBroker] Proposal failed for {order.symbol}: {e}")
            raise

        # 2. Buy
        try:
            buy_req = self.builder.buy(proposal_id, order.size)
            buy_resp = await self.client.send_request(buy_req)

            contract_id = str(buy_resp["buy"]["contract_id"])
            buy_price = float(buy_resp["buy"]["buy_price"])
        except Exception as e:
            logger.error(f"[LiveBroker] Buy failed for {order.symbol}: {e}")
            raise

        logger.info(
            f"[LiveBroker] LIVE OPENED {order.direction.name} {order.symbol} | "
            f"contract_id={contract_id} buy_price={buy_price}"
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
        # Write to the private dict inherited from PaperBroker
        self._open_positions[pos.id] = pos
        return pos

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
        current_time_ms: int | None = None,
    ) -> Trade | None:
        """
        Execute a live sell order.

        Checks is_valid_to_sell before attempting the sell so we don't get
        SellNotAvailable errors on contracts that haven't started trading or
        have already settled.
        """
        if position_id not in self._open_positions:
            return None

        pos = self._open_positions[position_id]

        # Check whether the contract is currently sellable
        try:
            poc_req = {
                "proposal_open_contract": 1,
                "contract_id": int(position_id),
                "req_id": self.builder._next_req_id(),
            }
            poc_resp = await self.client.send_request(poc_req)
            contract_info = poc_resp.get("proposal_open_contract", {})
            is_valid_to_sell = contract_info.get("is_valid_to_sell", 0)
            is_sold = contract_info.get("is_sold", 0)

            if is_sold:
                # Already settled on the server — remove locally and record
                logger.info(
                    f"[LiveBroker] Contract {position_id} already settled server-side. "
                    f"Recording as closed."
                )
                self._open_positions.pop(position_id, None)
                profit = float(contract_info.get("profit", 0.0))
                realized_pnl = profit
                realized_pnl_pct = profit / pos.entry_price if pos.entry_price else 0.0
                trade = Trade(
                    id=pos.id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=float(contract_info.get("current_spot", exit_price)),
                    realized_pnl=realized_pnl,
                    realized_pnl_pct=realized_pnl_pct,
                    opened_at=pos.opened_at,
                    closed_at=current_time_ms or int(time.time() * 1000),
                    close_reason="settled",
                    strategy_name=pos.strategy_name,
                )
                self.trade_history.append(trade)
                await self._fire_trade_callbacks(trade)
                return trade

            if not is_valid_to_sell:
                logger.warning(
                    f"[LiveBroker] Contract {position_id} is not currently sellable "
                    f"(is_valid_to_sell=0, reason={reason}). Skipping close attempt."
                )
                return None

        except Exception as e:
            logger.warning(
                f"[LiveBroker] Could not check is_valid_to_sell for {position_id}: {e}. "
                f"Attempting sell anyway."
            )

        # Sell
        try:
            sell_req = self.builder.sell(int(position_id))
            sell_resp = await self.client.send_request(sell_req)
            sold_for = float(sell_resp["sell"]["sold_for"])
        except Exception as e:
            logger.error(f"[LiveBroker] Sell failed for contract {position_id}: {e}")
            raise

        self._open_positions.pop(position_id)

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
        # Append to inherited trade_history list (PaperBroker doesn't have this;
        # we add it here for LiveBroker's own record-keeping)
        if not hasattr(self, "trade_history"):
            self.trade_history = []
        self.trade_history.append(trade)

        logger.info(
            f"[LiveBroker] LIVE CLOSED {pos.direction.name} {pos.symbol} | "
            f"contract_id={position_id} sold_for={sold_for:.2f} "
            f"PnL={realized_pnl:+.2f} reason={reason}"
        )

        await self._fire_trade_callbacks(trade)
        return trade

    async def _fire_trade_callbacks(self, trade: Trade) -> None:
        """Fire all registered trade handlers. Mirrors PaperBroker callback loop."""
        for handler in self._trade_handlers:
            await handler(trade)
