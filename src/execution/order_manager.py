"""
OrderManager — the gatekeeper between signals and execution.

Responsibilities:
1. IDEMPOTENCY — check if order_id already exists in DB before calling broker
2. ROUTING     — delegate to the injected BrokerInterface (paper or live)
3. PERSISTENCE — write every order state change to PostgreSQL

Why keep these three concerns in one class?
Because the sequence matters: idempotency check MUST happen before broker call,
and DB write MUST happen after. Separating them would require careful orchestration
anyway, so keeping them together is cleaner for Sprint 4 scope.

Why NOT write to DB from PaperBroker?
Single responsibility. The broker knows how to fill orders; it shouldn't know
about our DB schema. If we change the schema (Sprint 5), we only touch this file.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.execution.broker import BrokerInterface, Order, OrderSide, OrderStatus, RiskDecision

logger = logging.getLogger(__name__)

# Sprint 4 uses SQLite so we can run without a PostgreSQL server.
# Sprint 5 will swap this for a real PostgreSQL connection string.
# The schema (schema.sql) is written for PostgreSQL — SQLite is schema-compatible
# for the fields we use (SQLite treats all types as TEXT/REAL/INTEGER).
DEFAULT_DB_PATH = Path("data/trades.db")


class OrderManager:
    """Routes orders through idempotency check → broker → DB persistence.

    Parameters
    ----------
    broker  : Any BrokerInterface implementation (PaperBroker, UpstoxBroker, etc.)
    db_path : Path to SQLite DB file. Defaults to data/trades.db.
              In tests, pass ':memory:' for an in-memory DB.

    Usage
    -----
    manager = OrderManager(broker=PaperBroker(portfolio), db_path=':memory:')
    order = manager.submit(risk_decision, order_id="abc-123")
    # Submitting the same order_id again returns the existing order unchanged:
    same_order = manager.submit(risk_decision, order_id="abc-123")
    assert order.fill_price == same_order.fill_price
    """

    def __init__(
        self,
        broker: BrokerInterface,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self._broker = broker
        self._db_path = str(db_path)
        # For :memory: databases, sqlite3 creates a brand-new empty DB on every
        # connect() call. We keep one persistent connection so the table created
        # in _init_db() is visible to all subsequent operations.
        self._memory_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, decision: RiskDecision, order_id: str | None = None) -> Order:
        """Submit an order, enforcing idempotency.

        If order_id already exists in the DB, the existing order is returned
        immediately — the broker is never called a second time.

        If order_id is None, a new UUID is generated. This is the normal path
        when the user clicks "Paper Trade" without a pre-set order_id.

        Parameters
        ----------
        decision : Validated RiskDecision from the risk engine
        order_id : Optional pre-set ID for idempotency replay
        """
        order = Order.from_risk_decision(decision, order_id=order_id)

        # IDEMPOTENCY CHECK — the most important line in this file
        existing = self._load_order(order.order_id)
        if existing is not None:
            logger.info(
                "Idempotency: order %s already exists (status=%s) — returning without re-submitting",
                order.order_id,
                existing.status,
            )
            return existing

        # Persist PENDING state before calling broker.
        # If the app crashes after this line but before broker returns,
        # we have a record we can reconcile on restart.
        self._save_order(order)

        # Delegate to broker (paper fill or live execution)
        filled_order = self._broker.submit(order)

        # Persist the updated state (FILLED or REJECTED)
        self._update_order(filled_order)

        return filled_order

    def close_position(self, order_id: str, reason: str = "manual") -> Order | None:
        """Close an open position by order_id.

        Returns None if order_id not found or position is not open.
        """
        order = self._load_order(order_id)
        if order is None:
            logger.warning("close_position: order %s not found", order_id)
            return None
        if not order.is_open:
            logger.warning(
                "close_position: order %s is not open (status=%s)", order_id, order.status
            )
            return order

        closed_order = self._broker.close_position(order, reason=reason)
        self._update_order(closed_order)
        return closed_order

    def get_open_positions(self) -> list[Order]:
        """Return all orders with status=FILLED."""
        return self._query_orders_by_status(OrderStatus.FILLED)

    def get_all_orders(self) -> list[Order]:
        """Return all orders regardless of status."""
        with self._db_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    # ------------------------------------------------------------------
    # DB layer (SQLite for Sprint 4; swap connection string for PostgreSQL)
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the trades table if it doesn't exist."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._db_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    order_id        TEXT PRIMARY KEY,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    shares          INTEGER NOT NULL,
                    strategy        TEXT,
                    agent_reasoning TEXT,
                    confidence      REAL,

                    requested_price REAL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    capital_at_risk REAL,

                    status          TEXT NOT NULL,
                    created_at      TEXT NOT NULL,

                    fill_price      REAL,
                    fill_timestamp  TEXT,
                    slippage        REAL,

                    exit_price      REAL,
                    exit_timestamp  TEXT,
                    exit_reason     TEXT,
                    pnl             REAL,
                    pnl_pct         REAL,
                    r_multiple      REAL
                )
            """)
            conn.commit()

    @contextmanager
    def _db_conn(self) -> Generator[sqlite3.Connection, None, None]:
        # :memory: — reuse the single persistent connection so the schema survives
        if self._memory_conn is not None:
            yield self._memory_conn
            return
        # File-based — open and close per operation (safe for concurrent access)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _save_order(self, order: Order) -> None:
        with self._db_conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    order_id, symbol, side, shares, strategy, agent_reasoning, confidence,
                    requested_price, stop_loss, take_profit, capital_at_risk,
                    status, created_at,
                    fill_price, fill_timestamp, slippage,
                    exit_price, exit_timestamp, exit_reason,
                    pnl, pnl_pct, r_multiple
                ) VALUES (
                    :order_id, :symbol, :side, :shares, :strategy, :agent_reasoning, :confidence,
                    :requested_price, :stop_loss, :take_profit, :capital_at_risk,
                    :status, :created_at,
                    :fill_price, :fill_timestamp, :slippage,
                    :exit_price, :exit_timestamp, :exit_reason,
                    :pnl, :pnl_pct, :r_multiple
                )
                """,
                self._order_to_dict(order),
            )
            conn.commit()

    def _update_order(self, order: Order) -> None:
        with self._db_conn() as conn:
            conn.execute(
                """
                UPDATE trades SET
                    status          = :status,
                    fill_price      = :fill_price,
                    fill_timestamp  = :fill_timestamp,
                    slippage        = :slippage,
                    exit_price      = :exit_price,
                    exit_timestamp  = :exit_timestamp,
                    exit_reason     = :exit_reason,
                    pnl             = :pnl,
                    pnl_pct         = :pnl_pct,
                    r_multiple      = :r_multiple
                WHERE order_id = :order_id
                """,
                self._order_to_dict(order),
            )
            conn.commit()

    def _load_order(self, order_id: str) -> Order | None:
        with self._db_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE order_id = ?", (order_id,)
            ).fetchone()
        return self._row_to_order(row) if row else None

    def _query_orders_by_status(self, status: OrderStatus) -> list[Order]:
        with self._db_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _order_to_dict(order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "shares": order.shares,
            "strategy": order.strategy,
            "agent_reasoning": order.agent_reasoning,
            "confidence": order.confidence,
            "requested_price": order.requested_price,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "capital_at_risk": order.capital_at_risk,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
            "fill_price": order.fill_price,
            "fill_timestamp": order.fill_timestamp.isoformat() if order.fill_timestamp else None,
            "slippage": order.slippage,
            "exit_price": order.exit_price,
            "exit_timestamp": order.exit_timestamp.isoformat() if order.exit_timestamp else None,
            "exit_reason": order.exit_reason,
            "pnl": order.pnl,
            "pnl_pct": order.pnl_pct,
            "r_multiple": order.r_multiple,
        }

    @staticmethod
    def _row_to_order(row: sqlite3.Row) -> Order:
        def _dt(val: str | None) -> datetime | None:
            return datetime.fromisoformat(val) if val else None

        return Order(
            order_id=row["order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            shares=row["shares"],
            strategy=row["strategy"] or "",
            agent_reasoning=row["agent_reasoning"] or "",
            confidence=row["confidence"] or 0.0,
            requested_price=row["requested_price"] or 0.0,
            stop_loss=row["stop_loss"] or 0.0,
            take_profit=row["take_profit"] or 0.0,
            capital_at_risk=row["capital_at_risk"] or 0.0,
            status=OrderStatus(row["status"]),
            created_at=_dt(row["created_at"]) or datetime.utcnow(),
            fill_price=row["fill_price"],
            fill_timestamp=_dt(row["fill_timestamp"]),
            slippage=row["slippage"],
            exit_price=row["exit_price"],
            exit_timestamp=_dt(row["exit_timestamp"]),
            exit_reason=row["exit_reason"],
            pnl=row["pnl"],
            pnl_pct=row["pnl_pct"],
            r_multiple=row["r_multiple"],
        )
