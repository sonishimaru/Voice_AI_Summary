"""Episode segmentation: group a local day's utterances into contiguous episodes."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from .config import Config
from .db import transaction
from .timeutil import fmt_hm, local_day_bounds


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _fmt_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_episodes(conn: sqlite3.Connection, cfg: Config, day: str) -> list[int]:
    """(Re)build episodes for local date `day`, assigning `episode_id` on its utterances.

    Idempotent: any episodes already covering this day's utterances are deleted and
    rebuilt from scratch; other days are untouched. Returns the new episode ids, in
    chronological order.
    """
    start_utc, end_utc = local_day_bounds(day, cfg.summarize.timezone)

    rows = conn.execute(
        """
        SELECT u.id AS utterance_id, u.episode_id, u.t_start_ms, u.t_end_ms, u.speaker,
               r.source AS source, r.started_at_utc AS rec_started_at_utc
        FROM utterances u
        JOIN recordings r ON r.id = u.recording_id
        WHERE u.abs_start_utc >= ? AND u.abs_start_utc < ?
        ORDER BY u.abs_start_utc, u.t_start_ms
        """,
        (start_utc, end_utc),
    ).fetchall()

    old_episode_ids = sorted({r["episode_id"] for r in rows if r["episode_id"] is not None})

    gap = timedelta(minutes=cfg.episodes.gap_minutes)

    groups: list[dict] = []
    current: dict | None = None
    prev_end: datetime | None = None
    for row in rows:
        rec_started = _parse_iso(row["rec_started_at_utc"])
        abs_start = rec_started + timedelta(milliseconds=row["t_start_ms"])
        abs_end = rec_started + timedelta(milliseconds=row["t_end_ms"])

        if current is None or prev_end is None or (abs_start - prev_end) > gap:
            current = {
                "utterance_ids": [],
                "start": abs_start,
                "end": abs_end,
                "sources": set(),
                "speakers": set(),
            }
            groups.append(current)

        current["utterance_ids"].append(row["utterance_id"])
        current["end"] = max(current["end"], abs_end)
        current["sources"].add(row["source"])
        current["speakers"].add(row["speaker"])
        prev_end = current["end"]

    with transaction(conn):
        if old_episode_ids:
            placeholders = ",".join("?" for _ in old_episode_ids)
            conn.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", old_episode_ids)

        new_ids: list[int] = []
        for grp in groups:
            sources = grp["sources"]
            speakers = grp["speakers"]
            if "me" in speakers and "other" in speakers:
                kind = "call"
            elif speakers == {"other"}:
                kind = "media"  # system audio only: video / audio playback, not a conversation
            elif speakers == {"me"}:
                kind = "solo"
            else:
                kind = "ambient"
            source_mix = ",".join(sorted(sources))

            cur = conn.execute(
                """
                INSERT INTO episodes(started_at_utc, ended_at_utc, kind, title, source_mix)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (_fmt_iso(grp["start"]), _fmt_iso(grp["end"]), kind, source_mix),
            )
            episode_id = cur.lastrowid
            new_ids.append(episode_id)

            conn.executemany(
                "UPDATE utterances SET episode_id = ? WHERE id = ?",
                [(episode_id, uid) for uid in grp["utterance_ids"]],
            )

    return new_ids


def episode_transcript(conn: sqlite3.Connection, episode_id: int, tz: str) -> str:
    """Render an episode's utterances as `HH:MM [speaker] text` lines, oldest first."""
    rows = conn.execute(
        """
        SELECT abs_start_utc, speaker, text
        FROM utterances
        WHERE episode_id = ?
        ORDER BY abs_start_utc, t_start_ms
        """,
        (episode_id,),
    ).fetchall()
    lines = [f"{fmt_hm(row['abs_start_utc'], tz)} [{row['speaker']}] {row['text']}" for row in rows]
    return "\n".join(lines)
