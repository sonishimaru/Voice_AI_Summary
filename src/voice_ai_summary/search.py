"""Full-text search over utterances."""

from __future__ import annotations

import sqlite3

_SELECT = """
    SELECT u.abs_start_utc AS abs_start_utc, u.speaker AS speaker, u.text AS text,
           r.source AS source
    FROM utterances u
    JOIN recordings r ON r.id = u.recording_id
"""


def search(
    conn: sqlite3.Connection, query: str, *, limit: int = 50, day: str | None = None
) -> list[sqlite3.Row]:
    """Search utterance text. Trigram FTS for queries >= 3 chars, LIKE fallback otherwise."""
    day_clause = " AND u.abs_start_utc LIKE ?" if day else ""
    day_params = (f"{day}%",) if day else ()

    if len(query) >= 3:
        sql = (
            _SELECT
            + " JOIN utterances_fts ON utterances_fts.rowid = u.id"
            + f" WHERE utterances_fts MATCH ?{day_clause}"
            + " ORDER BY u.abs_start_utc LIMIT ?"
        )
        params = (query, *day_params, limit)
    else:
        sql = _SELECT + f" WHERE u.text LIKE ?{day_clause}" + " ORDER BY u.abs_start_utc LIMIT ?"
        params = (f"%{query}%", *day_params, limit)

    return conn.execute(sql, params).fetchall()
