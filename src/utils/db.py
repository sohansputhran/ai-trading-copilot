"""
Database connection factory for production-grade connection management.

This module implements the 12-factor app configuration pattern: all database
connection details come from the DATABASE_URL environment variable. The factory
automatically routes to SQLite (dev) or PostgreSQL (prod) based on the URL scheme.

Why this pattern:
- Single source of truth for DB connections
- Zero code changes between dev and prod environments
- Easy to test (just override DATABASE_URL)
- Secrets stay in env vars, never in code
"""

import os
import sqlite3
from typing import Any, Union

# PostgreSQL support is optional - only imported if DATABASE_URL is a postgres URL
try:
    import psycopg2
    from psycopg2.extensions import connection as PostgresConnection

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresConnection = Any  # Type hint fallback


def get_connection(
    database_url: str | None = None,
) -> Union[sqlite3.Connection, "PostgresConnection"]:
    """
    Create a database connection based on the connection URL.

    This factory function is the ONLY place in the codebase that decides which
    database driver to use. All other code imports this function and remains
    database-agnostic.

    Connection URL formats:
    - SQLite: "sqlite:///path/to/database.db" or "sqlite:///:memory:" or None (default)
    - PostgreSQL: "postgresql://user:password@host:port/dbname"

    Args:
        database_url: Connection string. If None, reads from DATABASE_URL env var.
                     If DATABASE_URL is unset, defaults to SQLite at data/trades.db.

    Returns:
        Either a sqlite3.Connection or psycopg2 connection object.
        Both share the same .cursor(), .commit(), .close() interface.

    Raises:
        ImportError: If PostgreSQL URL is used but psycopg2 is not installed.
        ValueError: If URL scheme is not recognized.

    Example:
        >>> # Dev mode (default)
        >>> conn = get_connection()  # Uses SQLite at data/trades.db

        >>> # Prod mode (via environment)
        >>> os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/trading_db"
        >>> conn = get_connection()  # Uses PostgreSQL

        >>> # Test mode (explicit override)
        >>> conn = get_connection("sqlite:///:memory:")  # In-memory SQLite
    """
    # Configuration: DATABASE_URL from env or explicit parameter
    url = database_url or os.getenv("DATABASE_URL", "sqlite:///data/trades.db")

    # Route based on URL scheme
    if url.startswith("sqlite"):
        return _connect_sqlite(url)
    elif url.startswith("postgresql") or url.startswith("postgres"):
        return _connect_postgresql(url)
    else:
        raise ValueError(
            f"Unsupported database URL scheme: {url}. " "Expected 'sqlite:///' or 'postgresql://'"
        )


def _connect_sqlite(url: str) -> sqlite3.Connection:
    """
    Create a SQLite connection from a sqlite:/// URL.

    SQLite URLs:
    - sqlite:///data/trades.db → File at data/trades.db (relative to CWD)
    - sqlite:////absolute/path/trades.db → Absolute file path
    - sqlite:///:memory: → In-memory database (tests)

    Args:
        url: SQLite connection URL

    Returns:
        sqlite3.Connection with row_factory set to Row (dict-like access)
    """
    # Strip "sqlite:///" prefix to get file path
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
    else:
        # Fallback for sqlite:// (two slashes) - some tools use this
        path = url.replace("sqlite://", "", 1)

    # Special case: :memory: for in-memory databases (testing)
    if path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        # Ensure parent directory exists for file-based SQLite
        if path != ":memory:":
            import pathlib

            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(path)

    # Enable dict-like row access: row["column_name"] instead of row[0]
    conn.row_factory = sqlite3.Row

    return conn


def _connect_postgresql(url: str) -> "PostgresConnection":
    """
    Create a PostgreSQL connection from a postgresql:// URL.

    PostgreSQL URL format:
    postgresql://username:password@hostname:port/database_name

    Args:
        url: PostgreSQL connection URL

    Returns:
        psycopg2 connection object

    Raises:
        ImportError: If psycopg2 is not installed
    """
    if not POSTGRES_AVAILABLE:
        raise ImportError(
            "PostgreSQL support requires psycopg2. Install with: pip install psycopg2-binary"
        )

    # psycopg2 accepts the full URL as-is
    conn = psycopg2.connect(url)

    # Enable autocommit for PostgreSQL (matches SQLite behavior)
    # Without this, you'd need explicit conn.commit() after every write
    conn.autocommit = False  # Keep transactions explicit for safety

    return conn


# Utility function for testing: check what DB type is currently configured
def get_database_type() -> str:
    """
    Return the type of database currently configured.

    Returns:
        "sqlite" or "postgresql"

    Example:
        >>> get_database_type()
        'sqlite'
        >>> os.environ["DATABASE_URL"] = "postgresql://..."
        >>> get_database_type()
        'postgresql'
    """
    url = os.getenv("DATABASE_URL", "sqlite:///data/trades.db")
    if url.startswith("sqlite"):
        return "sqlite"
    elif url.startswith("postgresql") or url.startswith("postgres"):
        return "postgresql"
    else:
        return "unknown"
