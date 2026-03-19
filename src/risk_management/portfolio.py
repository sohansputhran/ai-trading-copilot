"""
Portfolio Risk Aggregator

Tracks open positions and answers portfolio-level risk questions:
  - How much total capital is deployed?
  - How much is at risk right now?
  - What is the exposure to each sector?
  - What is today's P&L?
  - Do we have room for another trade?

Design: In-memory store for now. In future, I will add PostgreSQL persistence.
The interface is designed so callers don't need to know the backend.

Key concept — the difference between:
  - position_value:   how much it COSTS to hold the position (shares * price)
  - capital_at_risk:  how much you LOSE if stop is hit (shares * (entry - stop))

Never confuse these. A 50,000 position with a tight stop might only have
1,000 at risk. That's what risk management actually tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import structlog

logger = structlog.get_logger()

# Mirrors the hard limit in validators.py - single source of truth for the value
MAX_OPEN_POSITIONS = 5
MAX_DAILY_LOSS_PCT = 0.02


@dataclass
class Position:
    """
    A single open trade.

    Fields:
        symbol:          Stock symbol
        entry_price:     Price at which we entered
        stop_loss:       Exit price if trade goes against us
        shares:          Number of shares held
        position_value:  shares * entry_price (total cost)
        capital_at_risk: shares * (entry_price - stop_loss) (loss if stopped out)
        sector:          Optional sector label for exposure tracking
        entry_time:      When the trade was opened
        strategy:        Which agent/strategy generated this signal
    """

    symbol: str
    entry_price: float
    stop_loss: float
    shares: int
    position_value: float
    capital_at_risk: float
    sector: str | None = None
    entry_time: datetime = field(default_factory=datetime.now)
    strategy: str = "multi_agent"

    @property
    def risk_per_share(self) -> float:
        return self.entry_price - self.stop_loss

    def current_pnl(self, current_price: float) -> float:
        """Unrealized P&L at the given current price."""
        return (current_price - self.entry_price) * self.shares

    def current_value(self, current_price: float) -> float:
        """Current market value of the position."""
        return current_price * self.shares


@dataclass
class PortfolioSnapshot:
    """
    Point-in-time view of the portfolio's risk state.

    This is what the PreTradeValidator receives to make decisions.
    Returned by PortfolioRisk.snapshot().
    """

    portfolio_value: float
    open_positions: int
    total_position_value: float  # Sum of all position costs
    total_capital_at_risk: float  # Sum of all stop-loss risks
    daily_pnl: float  # Today's realized P&L
    sector_exposures: dict[str, float]  # {sector: total_position_value_in_sector}
    available_capital: float  # portfolio_value - total_position_value

    @property
    def total_deployed_pct(self) -> float:
        """Fraction of portfolio currently in positions."""
        if self.portfolio_value <= 0:
            return 0.0
        return self.total_position_value / self.portfolio_value

    @property
    def total_risk_pct(self) -> float:
        """Fraction of portfolio at risk (stop-loss basis)."""
        if self.portfolio_value <= 0:
            return 0.0
        return self.total_capital_at_risk / self.portfolio_value

    @property
    def daily_loss_pct(self) -> float:
        """Today's loss as a fraction of portfolio (positive = loss)."""
        if self.portfolio_value <= 0:
            return 0.0
        return max(0.0, -self.daily_pnl) / self.portfolio_value

    @property
    def circuit_breaker_triggered(self) -> bool:
        """True if daily loss has hit the 2% circuit breaker."""
        return self.daily_loss_pct >= MAX_DAILY_LOSS_PCT

    @property
    def positions_remaining(self) -> int:
        """How many more positions can be opened."""
        return max(0, MAX_OPEN_POSITIONS - self.open_positions)


class PortfolioRisk:
    """
    Tracks open positions and provides portfolio-level risk metrics.

    This is in-memory for now. In future, I will add PostgreSQL persistence.
    The interface is designed so callers don't need to know the backend.

    Usage:
        portfolio = PortfolioRisk(portfolio_value=500_000)

        # When a trade is approved and opened:
        position = Position(
            symbol="RELIANCE",
            entry_price=2500.0,
            stop_loss=2450.0,
            shares=40,
            position_value=100_000.0,
            capital_at_risk=2_000.0,
            sector="Energy",
        )
        portfolio.add_position(position)

        # Before the next trade:
        snap = portfolio.snapshot()
        print(snap.open_positions)       # 1
        print(snap.total_risk_pct)       # 0.004 (0.4%)
        print(snap.circuit_breaker_triggered)  # False

        # When a position is closed:
        portfolio.close_position("RELIANCE", exit_price=2580.0)
    """

    def __init__(self, portfolio_value: float) -> None:
        if portfolio_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")

        self.portfolio_value = portfolio_value
        self._positions: dict[str, Position] = {}  # symbol → Position
        self._daily_realized_pnl: float = 0.0
        self._trade_date: date = date.today()

        logger.info(
            "portfolio_risk_initialized",
            portfolio_value=portfolio_value,
        )

    # -------------------------------------------------------------------------
    # Position lifecycle
    # -------------------------------------------------------------------------

    def add_position(self, position: Position) -> None:
        """
        Register a new open position.

        Raises ValueError if:
          - Symbol already has an open position (no averaging down for now)
          - Would exceed MAX_OPEN_POSITIONS
        """
        if position.symbol in self._positions:
            raise ValueError(
                f"Position already open for {position.symbol}. "
                f"Close it before opening a new one."
            )
        if len(self._positions) >= MAX_OPEN_POSITIONS:
            raise ValueError(
                f"Cannot open position: already at maximum {MAX_OPEN_POSITIONS} open positions."
            )

        self._positions[position.symbol] = position
        logger.info(
            "position_opened",
            symbol=position.symbol,
            shares=position.shares,
            entry_price=position.entry_price,
            capital_at_risk=position.capital_at_risk,
            open_positions=len(self._positions),
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close an open position and record realized P&L.

        Returns the realized P&L for this trade.
        Raises KeyError if position doesn't exist.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position found for {symbol}")

        position = self._positions[symbol]
        realized_pnl = position.current_pnl(exit_price)

        # Reset daily P&L tracker if it's a new trading day
        self._reset_daily_pnl_if_new_day()
        self._daily_realized_pnl += realized_pnl

        del self._positions[symbol]

        logger.info(
            "position_closed",
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            realized_pnl=round(realized_pnl, 2),
            daily_pnl=round(self._daily_realized_pnl, 2),
            open_positions=len(self._positions),
        )

        return realized_pnl

    def update_portfolio_value(self, new_value: float) -> None:
        """Update the portfolio value (e.g., end-of-day mark-to-market)."""
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        logger.info(
            "portfolio_value_updated",
            old_value=old_value,
            new_value=new_value,
            change=new_value - old_value,
        )

    # -------------------------------------------------------------------------
    # Read methods (no side effects)
    # -------------------------------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        """
        Return a point-in-time snapshot of portfolio risk state.
        This is what PreTradeValidator consumes.
        """
        self._reset_daily_pnl_if_new_day()

        total_position_value = sum(p.position_value for p in self._positions.values())
        total_capital_at_risk = sum(p.capital_at_risk for p in self._positions.values())

        # Build sector exposure map
        sector_exposures: dict[str, float] = {}
        for pos in self._positions.values():
            if pos.sector:
                sector_exposures[pos.sector] = (
                    sector_exposures.get(pos.sector, 0.0) + pos.position_value
                )

        available_capital = self.portfolio_value - total_position_value

        snap = PortfolioSnapshot(
            portfolio_value=self.portfolio_value,
            open_positions=len(self._positions),
            total_position_value=total_position_value,
            total_capital_at_risk=total_capital_at_risk,
            daily_pnl=self._daily_realized_pnl,
            sector_exposures=sector_exposures,
            available_capital=available_capital,
        )

        logger.debug(
            "portfolio_snapshot",
            open_positions=snap.open_positions,
            total_deployed_pct=round(snap.total_deployed_pct * 100, 1),
            total_risk_pct=round(snap.total_risk_pct * 100, 2),
            daily_pnl=round(snap.daily_pnl, 2),
            circuit_breaker=snap.circuit_breaker_triggered,
        )

        return snap

    def get_sector_exposure(self, sector: str) -> float:
        """Return current portfolio exposure (in rupees) to the given sector."""
        return sum(p.position_value for p in self._positions.values() if p.sector == sector)

    def get_position(self, symbol: str) -> Position | None:
        """Return open position for symbol, or None."""
        return self._positions.get(symbol)

    def list_positions(self) -> list[Position]:
        """Return all open positions."""
        return list(self._positions.values())

    @property
    def open_position_count(self) -> int:
        return len(self._positions)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _reset_daily_pnl_if_new_day(self) -> None:
        """Reset daily P&L counter when the calendar date rolls over."""
        today = date.today()
        if today != self._trade_date:
            logger.info(
                "daily_pnl_reset",
                old_date=str(self._trade_date),
                new_date=str(today),
                final_daily_pnl=round(self._daily_realized_pnl, 2),
            )
            self._daily_realized_pnl = 0.0
            self._trade_date = today
