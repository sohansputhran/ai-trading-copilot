"""
Paper broker — simulates order execution against live Yahoo Finance prices.

Key design decisions:
- Re-fetches price at fill time (not the scanned price) because prices move
  between scan and execution. A scan might run at 10:30, user clicks "Paper Trade"
  at 10:35 — we fill at 10:35's price.
- Simulates slippage: real orders don't fill at the exact mid-price. Market orders
  slip to the ask (buys) or bid (sells). We model this with a configurable bps rate.
- Calls PortfolioRisk.add_position() after every fill so the risk sidebar stays accurate.
- Does NOT write to DB — that's OrderManager's job.
"""

from __future__ import annotations

import logging
from datetime import datetime

import yfinance as yf

from src.execution.broker import BrokerInterface, Order, OrderSide, OrderStatus
from src.risk_management.portfolio import PortfolioRisk, Position

logger = logging.getLogger(__name__)

# Default slippage: 5 basis points (0.05%) — a realistic estimate for
# liquid Indian large-caps. Increase for mid/small caps.
DEFAULT_SLIPPAGE_BPS = 5


class PaperBroker(BrokerInterface):
    """Simulates a broker using live Yahoo Finance prices.

    Parameters
    ----------
    portfolio   : PortfolioRisk instance shared with the risk sidebar.
                  After a fill, add_position() is called so the sidebar reflects
                  the new position immediately.
    slippage_bps: Basis points of slippage to apply. BUY orders fill slightly
                  above mid; SELL orders fill slightly below mid. Default: 5 bps.
    """

    def __init__(
        self,
        portfolio: PortfolioRisk,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    ) -> None:
        self._portfolio = portfolio
        self._slippage_factor = slippage_bps / 10_000  # convert bps → decimal

    # ------------------------------------------------------------------
    # Public interface (implements BrokerInterface)
    # ------------------------------------------------------------------

    def submit(self, order: Order) -> Order:
        """Fill a paper order at the current live price.

        Steps:
        1. Fetch live price from Yahoo Finance
        2. Apply slippage (buys pay slightly more, sells receive slightly less)
        3. Mark order FILLED
        4. Notify PortfolioRisk so sidebar stays in sync
        """
        if order.status != OrderStatus.PENDING:
            logger.warning(
                "submit() called on non-PENDING order %s (status=%s) — ignoring",
                order.order_id,
                order.status,
            )
            return order

        live_price = self._fetch_live_price(order.symbol)
        if live_price is None:
            logger.error("Could not fetch live price for %s — rejecting order", order.symbol)
            order.status = OrderStatus.REJECTED
            return order

        fill_price = self._apply_slippage(live_price, order.side)

        order.fill_price = fill_price
        order.fill_timestamp = datetime.utcnow()
        order.slippage = fill_price - order.requested_price
        order.status = OrderStatus.FILLED

        # Notify the portfolio risk engine — this keeps the sidebar accurate.
        # PortfolioRisk.add_position() takes a Position dataclass, not keyword args.
        self._portfolio.add_position(
            Position(
                symbol=order.symbol,
                entry_price=fill_price,
                stop_loss=order.stop_loss,
                shares=order.shares,
                position_value=fill_price * order.shares,
                capital_at_risk=order.capital_at_risk,
                sector=self._infer_sector(order.symbol),
                strategy=order.strategy,
            )
        )

        logger.info(
            "Paper fill: %s %d × %s @ ₹%.2f (slippage ₹%.2f)",
            order.side.value,
            order.shares,
            order.symbol,
            fill_price,
            order.slippage,
        )
        return order

    def close_position(self, order: Order, reason: str = "manual") -> Order:
        """Close an open paper position at the current live price.

        Calculates P&L and R-multiple for trade journal (Sprint 5).

        R-multiple = pnl / capital_at_risk
          +1R means the trade made exactly what was risked (ideal risk:reward = 2R+)
          -1R means the stop-loss was hit exactly
        """
        if order.status != OrderStatus.FILLED:
            logger.warning(
                "close_position() called on non-FILLED order %s (status=%s) — ignoring",
                order.order_id,
                order.status,
            )
            return order

        live_price = self._fetch_live_price(order.symbol)
        if live_price is None:
            logger.error("Could not fetch live price for %s — cannot close position", order.symbol)
            return order

        exit_price = self._apply_slippage(live_price, OrderSide.SELL)  # always exit via sell

        gross_pnl = (exit_price - order.fill_price) * order.shares  # type: ignore[operator]
        pnl_pct = (
            ((exit_price - order.fill_price) / order.fill_price) * 100  # type: ignore[operator]
            if order.fill_price
            else 0.0
        )
        r_multiple = gross_pnl / order.capital_at_risk if order.capital_at_risk > 0 else 0.0

        order.exit_price = exit_price
        order.exit_timestamp = datetime.utcnow()
        order.exit_reason = reason
        order.pnl = round(gross_pnl, 2)
        order.pnl_pct = round(pnl_pct, 2)
        order.r_multiple = round(r_multiple, 2)
        order.status = OrderStatus.STOPPED_OUT if reason == "stop_loss" else OrderStatus.CLOSED

        # Remove from portfolio risk tracking.
        # PortfolioRisk.close_position() requires exit_price to record realized P&L.
        self._portfolio.close_position(order.symbol, exit_price)

        logger.info(
            "Position closed: %s @ ₹%.2f | P&L: ₹%.2f (%.2f%%) | R: %.2fR",
            order.symbol,
            exit_price,
            order.pnl,
            order.pnl_pct,
            order.r_multiple,
        )
        return order

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_live_price(self, symbol: str) -> float | None:
        """Fetch the most recent close price from Yahoo Finance.

        Uses period='1d' for speed — this gives us the latest available price
        without downloading a full history.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.error("yfinance error fetching %s: %s", symbol, exc)
            return None

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Add slippage in the adverse direction.

        BUY  → pay more  (price × (1 + slippage))
        SELL → receive less (price × (1 - slippage))
        """
        if side == OrderSide.BUY:
            return round(price * (1 + self._slippage_factor), 2)
        return round(price * (1 - self._slippage_factor), 2)

    def _infer_sector(self, symbol: str) -> str:
        """Best-effort sector label for portfolio exposure tracking.

        A real implementation would look this up from a reference table.
        For now, we return 'Unknown' — good enough for Sprint 4 exposure bars.
        """
        # TODO Sprint 5: load from a symbol→sector mapping file
        return "Unknown"
