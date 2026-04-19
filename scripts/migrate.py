#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script

Migrates historical trade data from SQLite (dev) to PostgreSQL (production).
Designed to be idempotent: safe to run multiple times without duplicating data.

Usage:
    # Basic usage (reads DATABASE_URL from environment)
    python scripts/migrate.py

    # Explicit database URLs
    python scripts/migrate.py --sqlite data/trades.db --postgres postgresql://user:pass@localhost/trading_db

    # Dry-run mode (show what would be migrated without actually doing it)
    python scripts/migrate.py --dry-run

Requirements:
    - SQLite database with trades (from Sprints 4-5)
    - PostgreSQL database with schema initialized (see schema.sql)
    - Both sqlite3 and psycopg2 installed

Why Idempotent:
    Uses PostgreSQL's ON CONFLICT (order_id) DO NOTHING clause.
    If a trade with the same order_id already exists, it's skipped.
    This means you can:
    - Re-run the migration after adding more trades to SQLite
    - Recover from partial migrations (e.g., if script crashes mid-way)
    - Safely test in development without fear of duplicate data
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

from dotenv import load_dotenv

# Load .env file so DATABASE_URL is available without manual export
# override=True ensures .env values always win over stale system env vars
load_dotenv(override=True)

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.db import get_connection, get_database_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate trade data from SQLite to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use environment DATABASE_URL for target
  python scripts/migrate.py

  # Explicit source and target
  python scripts/migrate.py --sqlite data/trades.db --postgres postgresql://user:pass@host/db

  # Dry-run mode
  python scripts/migrate.py --dry-run
        """,
    )
    parser.add_argument(
        "--sqlite",
        default="data/trades.db",
        help="Path to source SQLite database (default: data/trades.db)",
    )
    parser.add_argument(
        "--postgres",
        default=None,
        help="PostgreSQL connection URL (default: read from DATABASE_URL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating",
    )
    return parser.parse_args()


def fetch_closed_trades(sqlite_path: str) -> List[Tuple]:
    """
    Fetch all closed/stopped_out trades from SQLite.

    Args:
        sqlite_path: Path to SQLite database file

    Returns:
        List of tuples, each containing all columns for one trade

    Raises:
        FileNotFoundError: If SQLite database doesn't exist
    """
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    logger.info(f"Reading trades from SQLite: {sqlite_path}")

    # Connect to SQLite (using our connection factory)
    conn = get_connection(f"sqlite:///{sqlite_path}")
    cursor = conn.cursor()

    # Query: Only migrate completed trades from the local SQLite trades table
    # (OrderManager writes to 'trades', not 'orders')
    query = """
        SELECT
            order_id,
            symbol,
            side,
            shares,
            requested_price,
            stop_loss,
            take_profit,
            capital_at_risk,
            strategy,
            agent_reasoning,
            confidence,
            status,
            created_at,
            fill_price,
            fill_timestamp,
            slippage,
            exit_price,
            exit_timestamp,
            exit_reason,
            pnl,
            pnl_pct,
            r_multiple
        FROM trades
        WHERE status IN ('CLOSED', 'STOPPED_OUT')
        ORDER BY created_at ASC
    """

    cursor.execute(query)
    trades = cursor.fetchall()

    conn.close()

    logger.info(f"Found {len(trades)} closed trades in SQLite")
    return trades


def migrate_to_postgresql(trades: List[Tuple], postgres_url: str, dry_run: bool = False):
    """
    Insert trades into PostgreSQL database.

    Uses ON CONFLICT (order_id) DO NOTHING for idempotency.
    This means:
    - Trades already in PostgreSQL are skipped (no duplicates)
    - Only new trades are inserted
    - Safe to run multiple times

    Args:
        trades: List of trade tuples from SQLite
        postgres_url: PostgreSQL connection URL
        dry_run: If True, log what would be inserted without actually inserting

    Returns:
        Tuple of (inserted_count, skipped_count)
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would migrate {len(trades)} trades to PostgreSQL")
        logger.info(f"[DRY RUN] Target: {postgres_url.split('@')[-1]}")  # Hide credentials
        return len(trades), 0

    logger.info(f"Connecting to PostgreSQL: {postgres_url.split('@')[-1]}")
    conn = get_connection(postgres_url)
    cursor = conn.cursor()

    # Insert query with ON CONFLICT clause for idempotency
    insert_query = """
        INSERT INTO trades (
            order_id, symbol, side, shares,
            requested_price, stop_loss, take_profit, capital_at_risk,
            strategy, agent_reasoning, confidence,
            status, created_at,
            fill_price, fill_timestamp, slippage,
            exit_price, exit_timestamp, exit_reason,
            pnl, pnl_pct, r_multiple
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (order_id) DO NOTHING
    """

    inserted_count = 0
    skipped_count = 0

    try:
        for trade in trades:
            cursor.execute(insert_query, trade)
            # rowcount = 0 means ON CONFLICT triggered (duplicate order_id)
            # rowcount = 1 means new row inserted
            if cursor.rowcount == 1:
                inserted_count += 1
                logger.debug(f"Inserted trade: {trade[0]} ({trade[1]})")  # order_id, symbol
            else:
                skipped_count += 1
                logger.debug(f"Skipped duplicate: {trade[0]}")

        conn.commit()
        logger.info(f"Migration complete: {inserted_count} inserted, {skipped_count} skipped")

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

    return inserted_count, skipped_count


def verify_migration(sqlite_path: str, postgres_url: str):
    """
    Verify migration by comparing record counts.

    Args:
        sqlite_path: Path to SQLite database
        postgres_url: PostgreSQL connection URL

    Returns:
        True if counts match, False otherwise
    """
    logger.info("Verifying migration...")

    # Count in SQLite
    sqlite_conn = get_connection(f"sqlite:///{sqlite_path}")
    sqlite_cursor = sqlite_conn.cursor()
    sqlite_cursor.execute("SELECT COUNT(*) FROM trades WHERE status IN ('CLOSED', 'STOPPED_OUT')")
    sqlite_count = sqlite_cursor.fetchone()[0]
    sqlite_conn.close()

    # Count in PostgreSQL
    postgres_conn = get_connection(postgres_url)
    postgres_cursor = postgres_conn.cursor()
    postgres_cursor.execute("SELECT COUNT(*) FROM trades WHERE status IN ('CLOSED', 'STOPPED_OUT')")
    postgres_count = postgres_cursor.fetchone()[0]
    postgres_conn.close()

    logger.info(f"SQLite closed trades: {sqlite_count}")
    logger.info(f"PostgreSQL trades: {postgres_count}")

    if sqlite_count == postgres_count:
        logger.info("✅ Verification passed: counts match")
        return True
    else:
        logger.warning(f"⚠️  Count mismatch: SQLite={sqlite_count}, PostgreSQL={postgres_count}")
        logger.warning("This may be expected if you've run the migration multiple times")
        logger.warning("or if some trades were already in PostgreSQL")
        return False


def main():
    """Main migration workflow."""
    args = parse_args()

    # Determine PostgreSQL URL
    postgres_url = args.postgres or os.getenv("DATABASE_URL")

    # Dry-run doesn't connect to PostgreSQL — only reads SQLite and reports
    if not args.dry_run:
        if not postgres_url or postgres_url.startswith("sqlite"):
            logger.error(
                "PostgreSQL URL required. Set DATABASE_URL environment variable "
                "or use --postgres argument"
            )
            sys.exit(1)
    else:
        # In dry-run use a placeholder if no URL provided
        postgres_url = postgres_url or "postgresql://localhost/dry_run_placeholder"

    try:
        # Step 1: Fetch trades from SQLite
        trades = fetch_closed_trades(args.sqlite)

        if not trades:
            logger.warning("No closed trades found in SQLite. Nothing to migrate.")
            return

        # Step 2: Migrate to PostgreSQL
        inserted, skipped = migrate_to_postgresql(trades, postgres_url, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("[DRY RUN] No changes made. Remove --dry-run to perform actual migration.")
            return

        # Step 3: Verify migration
        verify_migration(args.sqlite, postgres_url)

        logger.info("=" * 60)
        logger.info("Migration Summary")
        logger.info("=" * 60)
        logger.info(f"Source: {args.sqlite}")
        logger.info(f"Target: {postgres_url.split('@')[-1]}")
        logger.info(f"Inserted: {inserted} trades")
        logger.info(f"Skipped: {skipped} trades (already existed)")
        logger.info(f"Total processed: {len(trades)} trades")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
