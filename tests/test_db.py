"""Tests for schema application and the schema migrations (v1 -> v2 -> v3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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
