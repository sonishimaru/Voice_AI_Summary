"""Tests for episode segmentation. Inserts recordings/utterances directly via SQL."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from voice_ai_summary.config import Config
from voice_ai_summary.episodes import build_episodes, episode_transcript


def _insert_recording(
    conn: sqlite3.Connection, *, source: str, started_at_utc: str, sha256: str
) -> int:
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at
        ) VALUES (?, 'dev1', ?, '+00:00', 1000, ?, 'store/x.wav', 'x.wav', ?, ?)
        """,
        (source, started_at_utc, sha256, started_at_utc, started_at_utc),
    )
    return cur.lastrowid


def _insert_utterance(
    conn: sqlite3.Connection,
    *,
    recording_id: int,
    rec_started_at_utc: str,
    t_start_ms: int,
    t_end_ms: int,
    speaker: str,
    text: str = "テスト発話",
) -> int:
    base = datetime.strptime(rec_started_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    abs_start = (base + timedelta(milliseconds=t_start_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        INSERT INTO utterances(
            recording_id, t_start_ms, t_end_ms, abs_start_utc, text, lang,
            asr_model, avg_logprob, speaker
        ) VALUES (?, ?, ?, ?, ?, 'ja', 'fake', -0.1, ?)
        """,
        (recording_id, t_start_ms, t_end_ms, abs_start, text, speaker),
    )
    return cur.lastrowid


def test_build_episodes_splits_on_gap_and_classifies_kind(vas) -> None:
    cfg: Config
    cfg, conn = vas
    day = "2026-09-15"
    started = "2026-09-15T00:00:00Z"

    rec_mic = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="a" * 64)
    rec_sys = _insert_recording(conn, source="mac_system", started_at_utc=started, sha256="b" * 64)
    conn.commit()

    # Group 1 (call): interleaved me/other across two recordings, 0-35s.
    for rec_id, offset, speaker in [
        (rec_mic, 0, "me"),
        (rec_sys, 10_000, "other"),
        (rec_mic, 20_000, "me"),
        (rec_sys, 30_000, "other"),
    ]:
        _insert_utterance(
            conn,
            recording_id=rec_id,
            rec_started_at_utc=started,
            t_start_ms=offset,
            t_end_ms=offset + 5_000,
            speaker=speaker,
        )

    # Gap of 10 minutes (> default 5 minute gap_minutes) before group 2.
    group2_start = 35_000 + 10 * 60_000
    for i in range(4):
        offset = group2_start + i * 10_000
        _insert_utterance(
            conn,
            recording_id=rec_mic,
            rec_started_at_utc=started,
            t_start_ms=offset,
            t_end_ms=offset + 5_000,
            speaker="me",
        )
    conn.commit()

    episode_ids = build_episodes(conn, cfg, day)
    assert len(episode_ids) == 2

    rows = conn.execute(
        "SELECT id, kind, source_mix FROM episodes ORDER BY started_at_utc"
    ).fetchall()
    assert [r["kind"] for r in rows] == ["call", "solo"]
    assert rows[0]["source_mix"] == "mac_mic,mac_system"
    assert rows[1]["source_mix"] == "mac_mic"

    # All 8 utterances got assigned to one of the two episodes.
    counts = conn.execute(
        "SELECT episode_id, COUNT(*) AS n FROM utterances GROUP BY episode_id ORDER BY episode_id"
    ).fetchall()
    assert sum(c["n"] for c in counts) == 8
    assert all(c["episode_id"] is not None for c in counts)

    transcript = episode_transcript(conn, rows[0]["id"], cfg.summarize.timezone)
    assert "[me]" in transcript
    assert "[other]" in transcript
    assert len(transcript.splitlines()) == 4


def test_build_episodes_is_idempotent(vas) -> None:
    cfg, conn = vas
    day = "2026-09-15"
    started = "2026-09-15T00:00:00Z"
    rec_mic = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="c" * 64)
    conn.commit()
    for i in range(3):
        _insert_utterance(
            conn,
            recording_id=rec_mic,
            rec_started_at_utc=started,
            t_start_ms=i * 1_000,
            t_end_ms=i * 1_000 + 500,
            speaker="me",
        )
    conn.commit()

    first = build_episodes(conn, cfg, day)
    second = build_episodes(conn, cfg, day)

    assert len(first) == len(second) == 1
    kinds = conn.execute("SELECT kind FROM episodes").fetchall()
    assert [r["kind"] for r in kinds] == ["solo"]
    utt_count = conn.execute(
        "SELECT COUNT(*) AS n FROM utterances WHERE episode_id IS NOT NULL"
    ).fetchone()["n"]
    assert utt_count == 3


def test_build_episodes_respects_local_day_boundary_for_timezone(vas) -> None:
    cfg, conn = vas
    assert cfg.summarize.timezone == "Asia/Tokyo"

    # 2026-09-14T16:00:00Z is 2026-09-15T01:00:00 JST -> belongs to local day 2026-09-15.
    started = "2026-09-14T16:00:00Z"
    rec = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="d" * 64)
    conn.commit()
    _insert_utterance(
        conn,
        recording_id=rec,
        rec_started_at_utc=started,
        t_start_ms=0,
        t_end_ms=1_000,
        speaker="me",
    )
    conn.commit()

    assert build_episodes(conn, cfg, "2026-09-14") == []
    episode_ids = build_episodes(conn, cfg, "2026-09-15")
    assert len(episode_ids) == 1

    row = conn.execute("SELECT episode_id FROM utterances").fetchone()
    assert row["episode_id"] == episode_ids[0]


def test_system_only_episode_is_media(vas) -> None:
    cfg, conn = vas
    started = "2026-09-15T03:14:00Z"
    rec = _insert_recording(conn, source="mac_system", started_at_utc=started, sha256="m" * 64)
    for i in range(3):
        _insert_utterance(
            conn,
            recording_id=rec,
            rec_started_at_utc=started,
            t_start_ms=i * 2_000,
            t_end_ms=i * 2_000 + 1_000,
            speaker="other",
        )
    conn.commit()

    build_episodes(conn, cfg, "2026-09-15")
    assert [r["kind"] for r in conn.execute("SELECT kind FROM episodes")] == ["media"]
