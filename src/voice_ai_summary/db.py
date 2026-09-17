"""SQLite access: connection factory, schema application, small helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 5


def utcnow_iso() -> str:
    """ISO 8601 UTC timestamp with second precision and a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Long enough to outlast another process's write burst, short enough to surface a
# genuine deadlock. Transcription itself must stay outside the write lock (pipeline.py).
BUSY_TIMEOUT_S = 30.0


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the database and apply the schema."""
    path = Path(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_S)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        # The worker, `vas reprocess` and a CLI call can all be open at once. WAL lets
        # readers through while one of them writes; the timeout covers the rest.
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_S * 1000)}")
    apply_schema(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    sql = resources.files("voice_ai_summary").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate_utterances_columns(conn)
    _migrate_recordings_columns(conn)
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


def _migrate_recordings_columns(conn: sqlite3.Connection) -> None:
    """v2 -> v3 -> v4 -> v5: add `claimed_at`, `processing_ms` and `audio_deleted_at`
    to `recordings` for DBs created before those columns existed.

    Fresh databases already get these columns from `schema.sql`, so this is a no-op there.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(recordings)")}
    if "claimed_at" not in cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN claimed_at TEXT")
    if "processing_ms" not in cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN processing_ms INTEGER")
    if "audio_deleted_at" not in cols:
        conn.execute("ALTER TABLE recordings ADD COLUMN audio_deleted_at TEXT")


def recent_realtime_factor(conn: sqlite3.Connection, limit: int = 20) -> float | None:
    """Aggregate realtime factor - total audio duration / total processing time - over
    the most recently processed recordings that have both `duration_ms` and
    `processing_ms` recorded (older databases, and any recording processed before this
    timing was added, have neither).

    A factor above 1 means decoding runs faster than realtime; below 1 means slower.
    Returns None when there is no timing data yet at all - callers must say so rather
    than inventing a number.
    """
    rows = conn.execute(
        "SELECT duration_ms, processing_ms FROM recordings"
        " WHERE processing_ms IS NOT NULL AND duration_ms IS NOT NULL"
        " ORDER BY processed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return None
    total_audio_ms = sum(r["duration_ms"] for r in rows)
    total_proc_ms = sum(r["processing_ms"] for r in rows)
    if total_proc_ms <= 0:
        return None
    return total_audio_ms / total_proc_ms


def average_processed_duration_ms(conn: sqlite3.Connection, limit: int = 50) -> float | None:
    """Average `duration_ms` over the most recently processed recordings.

    Used to estimate the audio length of *pending* recordings, whose own `duration_ms`
    is not known until `process_recording` sets it. None when nothing has been
    processed yet.
    """
    rows = conn.execute(
        "SELECT duration_ms FROM recordings WHERE duration_ms IS NOT NULL"
        " ORDER BY processed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return None
    return sum(r["duration_ms"] for r in rows) / len(rows)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
