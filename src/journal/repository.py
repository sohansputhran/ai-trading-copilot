"""
Trade repository — the single layer that touches the database.

Pattern: Abstract Base Class (same pattern as BrokerInterface in Sprint 4).
Analytics code imports TradeRepository and calls .get_closed_trades().
It never imports sqlite3 or psycopg2 directly.

Implementations:
  - SQLiteRepository   — reads from the Sprint 4 data/trades.db, no migration needed.
  - PostgreSQLRepository — reads from PostgreSQL via DATABASE_URL env var.

Switching from SQLite to PostgreSQL in production = swap the concrete class.
All analytics code remains unchanged.
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod

from src.journal.models import ClosedTrade

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TradeRepository(ABC):
    """Abstract interface for reading closed trade data.

    Any concrete repository must implement get_closed_trades().
    Analytics code depends only on this interface, never on implementations.
    """

    @abstractmethod
    def get_closed_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[ClosedTrade]:
        """Return all closed trades, optionally filtered.

        Args:
            symbol:   If provided, return only trades for this symbol.
            strategy: If provided, return only trades for this strategy.

        Returns:
            List of ClosedTrade read models, ordered by exit_timestamp ASC.
        """
        ...

    @abstractmethod
    def get_equity_curve(self) -> list[dict]:
        """Return daily cumulative P&L for the equity curve chart.

        Returns:
            List of dicts with keys: date (str), cumulative_pnl (float).
            Ordered by date ASC.
        """
        ...


# ---------------------------------------------------------------------------
# SQLite implementation — Sprint 4 compatible, no data migration needed
# ---------------------------------------------------------------------------

_CLOSED_TRADES_QUERY = """
SELECT
    order_id,
    symbol,
    strategy,
    fill_timestamp,
    exit_timestamp,
    fill_price AS entry_price,
    exit_price,
    shares,
    stop_loss,
    status,
    pnl,
    pnl_pct
FROM trades
WHERE status IN ('CLOSED', 'STOPPED_OUT')
  AND exit_timestamp IS NOT NULL
  AND exit_price IS NOT NULL
"""

_EQUITY_CURVE_QUERY = """
SELECT
    date(exit_timestamp) AS date,
    SUM(pnl) AS daily_pnl
FROM trades
WHERE status IN ('CLOSED', 'STOPPED_OUT')
  AND exit_timestamp IS NOT NULL
GROUP BY date(exit_timestamp)
ORDER BY date ASC
"""


class SQLiteRepository(TradeRepository):
    """Reads closed trades from the Sprint 4 SQLite database.

    No data migration required — reads the existing trades table directly.
    The Sprint 4 schema stores all required columns (stop_loss was added in
    OrderManager as part of the RiskDecision handoff).

    Args:
        db_path: Path to the SQLite database file. Defaults to data/trades.db.
                 Pass ":memory:" in tests (with an existing connection).
        connection: An existing sqlite3 connection. If provided, db_path is
                    ignored. Used in tests where the in-memory DB is created
                    externally and passed in.
    """

    def __init__(
        self,
        db_path: str = "data/trades.db",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self._db_path = db_path
        self._conn = connection  # injected connection (for :memory: tests)

    def get_closed_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[ClosedTrade]:
        """Fetch all closed trades from SQLite, optionally filtered."""
        query = _CLOSED_TRADES_QUERY
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        query += " ORDER BY exit_timestamp ASC"

        rows = self._execute(query, params)
        return [ClosedTrade.from_db_row(dict(row)) for row in rows]

    def get_equity_curve(self) -> list[dict]:
        """Return daily P&L with running cumulative total."""
        rows = self._execute(_EQUITY_CURVE_QUERY, [])
        cumulative = 0.0
        result = []
        for row in rows:
            cumulative += float(row["daily_pnl"])
            result.append(
                {
                    "date": row["date"],
                    "daily_pnl": float(row["daily_pnl"]),
                    "cumulative_pnl": round(cumulative, 2),
                }
            )
        return result

    # Private

    def _execute(self, query: str, params: list):
        """Execute a query on the injected or file-based connection."""
        if self._conn is not None:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute(query, params)
            return cursor.fetchall()

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# PostgreSQL implementation — ready for Sprint 5 / production migration
# ---------------------------------------------------------------------------

# PostgreSQL uses $1/$2 placeholders instead of SQLite's ?
_PG_CLOSED_TRADES_QUERY = _CLOSED_TRADES_QUERY.replace("?", "${}")
_PG_EQUITY_CURVE_QUERY = _EQUITY_CURVE_QUERY.replace("date(exit_timestamp)", "exit_timestamp::date")


class PostgreSQLRepository(TradeRepository):
    """Reads closed trades from PostgreSQL using the schema.sql DDL.

    Connection string must be provided via the DATABASE_URL environment
    variable — never hardcoded. Example:
        DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db

    Requires psycopg2-binary (not installed by default — add to requirements
    before activating this repository).

    Usage:
        repo = PostgreSQLRepository()        # reads DATABASE_URL from env
        trades = repo.get_closed_trades()
    """

    def __init__(self) -> None:
        self._dsn = os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise OSError(
                "DATABASE_URL environment variable is not set. "
                "Example: postgresql://user:pass@localhost:5432/trading_db"
            )

    def get_closed_trades(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[ClosedTrade]:
        """Fetch closed trades from PostgreSQL."""
        import psycopg2
        import psycopg2.extras

        query = _build_pg_query(_CLOSED_TRADES_QUERY, symbol, strategy)
        params = [p for p in [symbol, strategy] if p is not None]

        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params or None)
                rows = cur.fetchall()

        return [ClosedTrade.from_db_row(dict(row)) for row in rows]

    def get_equity_curve(self) -> list[dict]:
        """Return daily cumulative P&L from PostgreSQL."""
        import psycopg2
        import psycopg2.extras

        pg_query = """
        SELECT
            exit_timestamp::date AS date,
            SUM(pnl) AS daily_pnl
        FROM trades
        WHERE status IN ('CLOSED', 'STOPPED_OUT')
          AND exit_timestamp IS NOT NULL
        GROUP BY exit_timestamp::date
        ORDER BY date ASC
        """

        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(pg_query)
                rows = cur.fetchall()

        cumulative = 0.0
        result = []
        for row in rows:
            cumulative += float(row["daily_pnl"])
            result.append(
                {
                    "date": str(row["date"]),
                    "daily_pnl": float(row["daily_pnl"]),
                    "cumulative_pnl": round(cumulative, 2),
                }
            )
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_pg_query(base: str, symbol: str | None, strategy: str | None) -> str:
    """Build a parameterised PostgreSQL query with $1/$2 placeholders."""
    query = base
    idx = 1
    if symbol:
        query += f" AND symbol = ${idx}"
        idx += 1
    if strategy:
        query += f" AND strategy = ${idx}"
        idx += 1
    query += " ORDER BY exit_timestamp ASC"
    return query
