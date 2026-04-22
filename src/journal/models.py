"""
Trade journal read models.

These are *read models* — data shapes optimised for analytics queries.
They are distinct from the Order write model (src/execution/broker.py) which
tracks the full order lifecycle including pending/rejected states.

Design principle (CQRS-lite):
  - Write path: Order → OrderManager → DB
  - Read path:  DB → TradeRepository → ClosedTrade → PerformanceEngine

ClosedTrade only represents completed trades (CLOSED or STOPPED_OUT).
Derived fields (hold_days, r_multiple) are computed once at read time so
PerformanceEngine never has to re-derive them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClosedTrade:
    """Read model for a completed trade — used exclusively by analytics.

    Fields map to the `closed_trades` DB view defined in schema.sql.
    Derived fields are computed by TradeRepository at read time.

    Attributes:
        order_id:        Unique identifier (matches Order.order_id).
        symbol:          NSE ticker symbol (e.g. "RELIANCE").
        strategy:        Strategy tag from the originating signal.
        entry_timestamp: When the position was opened (FILLED time).
        exit_timestamp:  When the position was closed or stopped out.
        entry_price:     Fill price on entry.
        exit_price:      Fill price on exit.
        shares:          Number of shares traded.
        stop_loss:       Initial stop-loss price (used to compute 1R).
        status:          "CLOSED" (take-profit / manual) or "STOPPED_OUT".
        pnl:             Realised P&L in rupees (can be negative).
        pnl_pct:         Realised P&L as a percentage of entry position value.
        hold_days:       Calendar days between entry and exit (derived).
        r_multiple:      P&L expressed in units of initial risk (derived).
                         Formula: pnl / (entry_price - stop_loss) / shares
                         Positive = winner; negative = loser.
    """

    order_id: str
    symbol: str
    strategy: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    shares: int
    stop_loss: float
    status: str  # "CLOSED" | "STOPPED_OUT"
    pnl: float
    pnl_pct: float
    hold_days: int
    r_multiple: float

    @classmethod
    def from_db_row(cls, row: dict) -> ClosedTrade:
        """Construct a ClosedTrade from a raw DB row dict.

        Handles timestamp parsing and derives hold_days / r_multiple
        so downstream analytics code never touches raw DB types.

        Args:
            row: Dict with keys matching the closed_trades view columns.
        """
        entry_ts = _parse_timestamp(row["fill_timestamp"])
        exit_ts = _parse_timestamp(row["exit_timestamp"])

        hold_days = max(0, (exit_ts.date() - entry_ts.date()).days)

        # Initial risk per share = entry - stop_loss (always positive for long)
        initial_risk_per_share = row["entry_price"] - row["stop_loss"]
        if initial_risk_per_share > 0 and row["shares"] > 0:
            r_multiple = row["pnl"] / (initial_risk_per_share * row["shares"])
        else:
            r_multiple = 0.0

        return cls(
            order_id=row["order_id"],
            symbol=row["symbol"],
            strategy=row["strategy"],
            entry_timestamp=entry_ts,
            exit_timestamp=exit_ts,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            shares=int(row["shares"]),
            stop_loss=float(row["stop_loss"]),
            status=row["status"],
            pnl=float(row["pnl"]),
            pnl_pct=float(row["pnl_pct"]),
            hold_days=hold_days,
            r_multiple=round(r_multiple, 3),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value) -> datetime:
    """Parse a timestamp value that may be a string or already a datetime."""
    if isinstance(value, datetime):
        return value
    # SQLite stores datetimes as ISO strings
    # Try formats in order of likelihood (most common first)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO with microseconds: 2026-04-19T10:09:22.828670
        "%Y-%m-%d %H:%M:%S.%f",  # Space-separated with microseconds
        "%Y-%m-%dT%H:%M:%S",     # ISO without microseconds
        "%Y-%m-%d %H:%M:%S",     # Space-separated without microseconds
    ):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Cannot parse timestamp: {value!r}")
