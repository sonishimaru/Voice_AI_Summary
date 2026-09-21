"""Tests for schema application and the schema migrations (v1 -> v2 -> v3 -> v4 -> v5)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from voice_ai_summary.db import SCHEMA_VERSION, connect


def test_connect_creates_current_schema(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vas.sqlite3")
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(utterances)")}
    assert {"raw_text", "corrected_at", "correction_model"} <= cols
    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == SCHEMA_VERSION
    conn.close()


def test_migration_adds_correction_columns_to_v1_database(tmp_path: Path) -> None:
    """A DB created before the correction pass existed (v1: `schema_version` = 1, no
    `raw_text`/`corrected_at`/`correction_model` on `utterances`) gets those columns
    added, and its existing rows survive, the next time it's opened via `connect()`."""
    db_path = tmp_path / "vas.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version(version) VALUES (1);

        CREATE TABLE utterances (
            id             INTEGER PRIMARY KEY,
            recording_id   INTEGER NOT NULL,
            segment_id     INTEGER,
            episode_id     INTEGER,
            t_start_ms     INTEGER NOT NULL,
            t_end_ms       INTEGER NOT NULL,
            abs_start_utc  TEXT NOT NULL,
            text           TEXT NOT NULL,
            lang           TEXT,
            asr_model      TEXT,
            avg_logprob    REAL,
            speaker        TEXT NOT NULL DEFAULT 'unknown'
        );
        INSERT INTO utterances(
            id, recording_id, t_start_ms, t_end_ms, abs_start_utc, text, speaker
        ) VALUES (1, 1, 0, 500, '2026-09-15T00:00:00Z', 'テスト', 'me');
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = connect(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(utterances)")}
    assert {"raw_text", "corrected_at", "correction_model"} <= cols

    row = conn.execute("SELECT * FROM utterances WHERE id = 1").fetchone()
    assert row["text"] == "テスト"
    assert row["raw_text"] is None
    assert row["corrected_at"] is None

    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    # This DB predates both migrations (utterances correction columns and recordings
    # claimed_at), so connect() must run both and land it on the current version.
    assert version == SCHEMA_VERSION
    conn.close()


def test_migration_adds_claimed_at_to_v2_database(tmp_path: Path) -> None:
    """A DB created before the claim column existed (v2: `schema_version` = 2, no
    `claimed_at` on `recordings`) gets that column added, and its existing rows
    survive, the next time it's opened via `connect()`."""
    db_path = tmp_path / "vas.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version(version) VALUES (2);

        CREATE TABLE recordings (
            id              INTEGER PRIMARY KEY,
            source          TEXT NOT NULL,
            device_id       TEXT NOT NULL DEFAULT '',
            started_at_utc  TEXT NOT NULL,
            tz_offset       TEXT NOT NULL DEFAULT '+00:00',
            duration_ms     INTEGER,
            sha256          TEXT NOT NULL UNIQUE,
            storage_path    TEXT NOT NULL,
            original_name   TEXT NOT NULL DEFAULT '',
            ingested_at     TEXT NOT NULL,
            processed_at    TEXT,
            error           TEXT
        );
        INSERT INTO recordings(
            id, source, started_at_utc, sha256, storage_path, ingested_at
        ) VALUES (
            1, 'mac_mic', '2026-09-15T00:00:00Z', 'abc', 'x.wav', '2026-09-15T00:00:00Z'
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = connect(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(recordings)")}
    assert "claimed_at" in cols

    row = conn.execute("SELECT * FROM recordings WHERE id = 1").fetchone()
    assert row["storage_path"] == "x.wav"
    assert row["claimed_at"] is None

    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == SCHEMA_VERSION
    conn.close()


def test_migration_adds_processing_ms_to_v3_database(tmp_path: Path) -> None:
    """A DB created before `processing_ms` existed (v3: `schema_version` = 3, no
    `processing_ms` on `recordings`) gets that column added, and its existing rows -
    including a real user's 70+ already-transcribed recordings - survive untouched the
    next time it's opened via `connect()`."""
    db_path = tmp_path / "vas.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version(version) VALUES (3);

        CREATE TABLE recordings (
            id              INTEGER PRIMARY KEY,
            source          TEXT NOT NULL,
            device_id       TEXT NOT NULL DEFAULT '',
            started_at_utc  TEXT NOT NULL,
            tz_offset       TEXT NOT NULL DEFAULT '+00:00',
            duration_ms     INTEGER,
            sha256          TEXT NOT NULL UNIQUE,
            storage_path    TEXT NOT NULL,
            original_name   TEXT NOT NULL DEFAULT '',
            ingested_at     TEXT NOT NULL,
            processed_at    TEXT,
            error           TEXT,
            claimed_at      TEXT
        );
        INSERT INTO recordings(
            id, source, started_at_utc, duration_ms, sha256, storage_path, ingested_at,
            processed_at
        ) VALUES (
            1, 'mac_mic', '2026-09-15T00:00:00Z', 900000, 'abc', 'x.wav',
            '2026-09-15T00:00:00Z', '2026-09-15T00:05:00Z'
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = connect(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(recordings)")}
    assert "processing_ms" in cols

    row = conn.execute("SELECT * FROM recordings WHERE id = 1").fetchone()
    assert row["storage_path"] == "x.wav"
    assert row["duration_ms"] == 900000
    assert row["processing_ms"] is None

    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == SCHEMA_VERSION
    conn.close()


def test_migration_adds_audio_deleted_at_to_v4_database(tmp_path: Path) -> None:
    """A DB created before `audio_deleted_at` existed (v4: `schema_version` = 4, no
    `audio_deleted_at` on `recordings`) gets that column added, and its existing rows
    survive untouched the next time it's opened via `connect()`."""
    db_path = tmp_path / "vas.sqlite3"
    old_conn = sqlite3.connect(str(db_path))
    old_conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version(version) VALUES (4);

        CREATE TABLE recordings (
            id              INTEGER PRIMARY KEY,
            source          TEXT NOT NULL,
            device_id       TEXT NOT NULL DEFAULT '',
            started_at_utc  TEXT NOT NULL,
            tz_offset       TEXT NOT NULL DEFAULT '+00:00',
            duration_ms     INTEGER,
            sha256          TEXT NOT NULL UNIQUE,
            storage_path    TEXT NOT NULL,
            original_name   TEXT NOT NULL DEFAULT '',
            ingested_at     TEXT NOT NULL,
            processed_at    TEXT,
            error           TEXT,
            claimed_at      TEXT,
            processing_ms   INTEGER
        );
        INSERT INTO recordings(
            id, source, started_at_utc, duration_ms, sha256, storage_path, ingested_at,
            processed_at, processing_ms
        ) VALUES (
            1, 'mac_mic', '2026-09-15T00:00:00Z', 900000, 'abc', 'x.wav',
            '2026-09-15T00:00:00Z', '2026-09-15T00:05:00Z', 12000
        );
        """
    )
    old_conn.commit()
    old_conn.close()

    conn = connect(db_path)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(recordings)")}
    assert "audio_deleted_at" in cols

    row = conn.execute("SELECT * FROM recordings WHERE id = 1").fetchone()
    assert row["storage_path"] == "x.wav"
    assert row["processing_ms"] == 12000
    assert row["audio_deleted_at"] is None

    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == SCHEMA_VERSION
    conn.close()


def test_recent_realtime_factor_aggregates_over_recent_recordings(tmp_path: Path) -> None:
    from voice_ai_summary.db import recent_realtime_factor

    conn = connect(tmp_path / "vas.sqlite3")
    assert recent_realtime_factor(conn) is None  # no timing data yet

    def _insert(sha: str, duration_ms: int, processing_ms: int, processed_at: str) -> None:
        conn.execute(
            "INSERT INTO recordings(source, started_at_utc, duration_ms, sha256, "
            "storage_path, ingested_at, processed_at, processing_ms) VALUES "
            "('mac_mic', '2026-09-15T00:00:00Z', ?, ?, 'x.wav', '2026-09-15T00:00:00Z', ?, ?)",
            (duration_ms, sha, processed_at, processing_ms),
        )

    # 60s of audio in 30s -> 2x faster than realtime.
    _insert("a" * 64, 60_000, 30_000, "2026-09-15T00:01:00Z")
    # 60s of audio in 60s -> 1x realtime.
    _insert("b" * 64, 60_000, 60_000, "2026-09-15T00:02:00Z")
    conn.commit()

    # Total audio 120s, total processing 90s -> 120/90.
    assert recent_realtime_factor(conn) == pytest.approx(120_000 / 90_000)
    # limit=1 only picks up the most recently processed row (b).
    assert recent_realtime_factor(conn, limit=1) == pytest.approx(1.0)
    conn.close()


def test_average_processed_duration_ms(tmp_path: Path) -> None:
    from voice_ai_summary.db import average_processed_duration_ms

    conn = connect(tmp_path / "vas.sqlite3")
    assert average_processed_duration_ms(conn) is None

    conn.execute(
        "INSERT INTO recordings(source, started_at_utc, duration_ms, sha256, storage_path, "
        "ingested_at, processed_at) VALUES ('mac_mic', '2026-09-15T00:00:00Z', 60000, 'a', "
        "'x.wav', '2026-09-15T00:00:00Z', '2026-09-15T00:01:00Z')"
    )
    conn.execute(
        "INSERT INTO recordings(source, started_at_utc, duration_ms, sha256, storage_path, "
        "ingested_at, processed_at) VALUES ('mac_mic', '2026-09-15T00:00:00Z', 120000, 'b', "
        "'x.wav', '2026-09-15T00:00:00Z', '2026-09-15T00:02:00Z')"
    )
    conn.commit()

    assert average_processed_duration_ms(conn) == pytest.approx(90_000)
    conn.close()
