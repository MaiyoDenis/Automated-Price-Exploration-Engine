"""
Project APEX — Live Broker Stub

Real order execution via the Deriv API.

TODO(live-execution): This module is a stub. Real order submission requires
the Deriv API token with trading permissions. Implement once the user provides
the live API credentials.

The interface mirrors PaperBroker so the application can swap brokers without
changing the rest of the system.
"""
from __future__ import annotations

from loguru import logger

from project_apex.risk.models import TradeOrder
from project_apex.execution.models import Position, Trade
from project_apex.execution.paper_broker import PaperBroker, TradeHandler


class LiveBroker(PaperBroker):
    """
    Live order execution broker (Deriv API).

    Currently inherits PaperBroker behaviour and logs all orders as
    "would-be live orders". Replace the methods below with real Deriv
    API calls once credentials are available.

    The Deriv trading API endpoints needed:
      - buy: https://api.deriv.com/api-explorer/#buy
      - sell: https://api.deriv.com/api-explorer/#sell
      - proposal: https://api.deriv.com/api-explorer/#proposal

    Security: API token MUST be loaded from environment / secrets manager.
    NEVER hardcode the token. See project_apex/config/environment.py.
    """

    def __init__(self, slippage_pct: float = 0.0, spread_pct: float = 0.0) -> None:
        super().__init__(slippage_pct=slippage_pct, spread_pct=spread_pct)
        logger.warning(
            "[LiveBroker] Running in LIVE-STUB mode. "
            "Real Deriv API calls are NOT implemented yet. "
            "All trades are paper-simulated until API credentials are provided."
        )

    async def open_position(self, order: TradeOrder, current_time_ms: int | None = None) -> Position:
        """
        TODO(live-execution): Replace with real Deriv API buy call.

        Real implementation steps:
          1. Send `proposal` request to Deriv WS to get a price quote.
          2. Send `buy` request with the proposal id and stake.
          3. Parse the `buy` response to get the contract ID.
          4. Map contract ID to a Position object.
        """
        logger.warning(
            f"[LiveBroker] LIVE ORDER (stub) — {order.direction.name} "
            f"{order.symbol} size={order.size:.4f}"
        )
        return await super().open_position(order, current_time_ms)

    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
        current_time_ms: int | None = None,
    ) -> Trade | None:
        """
        TODO(live-execution): Replace with real Deriv API sell call.

        Real implementation steps:
          1. Look up the Deriv contract ID for this position_id.
          2. Send `sell` request with the contract ID.
          3. Parse the `sell` response for the final P&L.
        """
        logger.warning(
            f"[LiveBroker] LIVE CLOSE (stub) — position={position_id[:8]} reason={reason}"
        )
        return await super().close_position(position_id, exit_price, reason, current_time_ms)
