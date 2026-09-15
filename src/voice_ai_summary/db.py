"""SQLite access: connection factory, schema application, small helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 2


def utcnow_iso() -> str:
    """ISO 8601 UTC timestamp with second precision and a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the database and apply the schema."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    apply_schema(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    sql = resources.files("voice_ai_summary").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate_utterances_columns(conn)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] < SCHEMA_VERSION:
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    conn.commit()


def _migrate_utterances_columns(conn: sqlite3.Connection) -> None:
    """v1 -> v2: add the correction-pass columns to `utterances` for DBs created before them.

    Fresh databases already get these columns from `schema.sql`, so this is a no-op there.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(utterances)")}
    for name in ("raw_text", "corrected_at", "correction_model"):
        if name not in cols:
            conn.execute(f"ALTER TABLE utterances ADD COLUMN {name} TEXT")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
