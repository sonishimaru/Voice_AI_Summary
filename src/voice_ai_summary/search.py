"""Full-text search over utterances."""

from __future__ import annotations

import re
import sqlite3

from .timeutil import local_day_bounds

_SELECT = """
    SELECT u.abs_start_utc AS abs_start_utc, u.speaker AS speaker, u.text AS text,
           r.source AS source
    FROM utterances u
    JOIN recordings r ON r.id = u.recording_id
"""

# FTS5 treats punctuation as syntax, so a plain phrase like `A-1` or `"foo` is a
# query-language error rather than a search. Quoting every bareword token makes any
# user input a literal phrase search.
_FTS_TOKEN = re.compile(r'"[^"]*"|\S+')


def fts_query(query: str) -> str:
    """Escape `query` so FTS5 reads it as literal phrase(s), never as syntax."""
    tokens = []
    for token in _FTS_TOKEN.findall(query):
        inner = token[1:-1] if token.startswith('"') and token.endswith('"') else token
        tokens.append('"' + inner.replace('"', '""') + '"')
    return " ".join(tokens)


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
    day: str | None = None,
    tz: str = "UTC",
) -> list[sqlite3.Row]:
    """Search utterance text. Trigram FTS for queries >= 3 chars, LIKE fallback otherwise.

    `day` is a local date in `tz`, matched against the UTC instants it covers.
    """
    day_clause = ""
    day_params: tuple[str, ...] = ()
    if day:
        start, end = local_day_bounds(day, tz)
        day_clause = " AND u.abs_start_utc >= ? AND u.abs_start_utc < ?"
        day_params = (start, end)

    if len(query.strip()) >= 3:
        sql = (
            _SELECT
            + " JOIN utterances_fts ON utterances_fts.rowid = u.id"
            + f" WHERE utterances_fts MATCH ?{day_clause}"
            + " ORDER BY u.abs_start_utc LIMIT ?"
        )
        params = (fts_query(query), *day_params, limit)
    else:
        sql = _SELECT + f" WHERE u.text LIKE ?{day_clause}" + " ORDER BY u.abs_start_utc LIMIT ?"
        params = (f"%{query}%", *day_params, limit)

    return conn.execute(sql, params).fetchall()
