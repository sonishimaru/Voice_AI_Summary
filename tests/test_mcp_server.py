"""Tests for the MCP stdio server's tool functions.

Tool functions are called directly (module-level, `@_tool_safe`-wrapped callables) -
no MCP transport, no network, no Anthropic API calls.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_ai_summary import mcp_server
from voice_ai_summary.asr import FakeBackend
from voice_ai_summary.config import Config
from voice_ai_summary.summarize import PROMPT_VERSION

_UNSET = object()


def _insert_recording(
    conn: sqlite3.Connection,
    *,
    source: str,
    started_at_utc: str,
    sha256: str,
    duration_ms: int = 1000,
    processing_ms: int | None = None,
    processed_at: str | None = _UNSET,  # type: ignore[assignment]
) -> int:
    """`processed_at` defaults to `started_at_utc` (an already-processed recording);
    pass `processed_at=None` explicitly for a still-pending recording."""
    if processed_at is _UNSET:
        processed_at = started_at_utc
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at, processing_ms
        ) VALUES (?, 'dev1', ?, '+00:00', ?, ?, 'store/x.wav', 'x.wav', ?, ?, ?)
        """,
        (source, started_at_utc, duration_ms, sha256, started_at_utc, processed_at, processing_ms),
    )
    conn.commit()
    return cur.lastrowid


def _insert_utterance(
    conn: sqlite3.Connection,
    *,
    recording_id: int,
    abs_start_utc: str,
    t_start_ms: int = 0,
    speaker: str = "me",
    text: str = "テスト発話",
    asr_model: str = "fake",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO utterances(
            recording_id, t_start_ms, t_end_ms, abs_start_utc, text, lang,
            asr_model, avg_logprob, speaker
        ) VALUES (?, ?, ?, ?, ?, 'ja', ?, -0.1, ?)
        """,
        (recording_id, t_start_ms, t_start_ms + 500, abs_start_utc, text, asr_model, speaker),
    )
    conn.commit()
    return cur.lastrowid


def _insert_day_summary(conn: sqlite3.Connection, day: str, markdown: str) -> None:
    conn.execute(
        """
        INSERT INTO summaries(scope, scope_key, model, prompt_version, json, markdown, created_at)
        VALUES ('day', ?, 'test-model', ?, '{}', ?, ?)
        """,
        (day, PROMPT_VERSION, markdown, "2026-09-15T00:00:00Z"),
    )
    conn.commit()


class TestDailySummary:
    def test_returns_stored_markdown(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        day = "2026-09-15"
        _insert_day_summary(conn, day, f"# {day} の記録\n\nハイライトです。\n")

        result = mcp_server.daily_summary(day=day)
        assert result == f"# {day} の記録\n\nハイライトです。\n"

    def test_no_digest_gives_helpful_message(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        day = "2026-01-01"
        result = mcp_server.daily_summary(day=day)
        assert day in result
        assert "rebuild_day" in result


class TestSearchTranscript:
    def test_formats_local_times(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_mic", started_at_utc="2026-09-15T00:00:00Z", sha256="a" * 64
        )
        # 2026-09-15T01:23:00Z is 10:23 in the default Asia/Tokyo (+09:00) timezone.
        _insert_utterance(
            conn,
            recording_id=rec,
            abs_start_utc="2026-09-15T01:23:00Z",
            speaker="other",
            text="会議の予定について話しました",
        )

        result = mcp_server.search_transcript(query="会議の予定")
        assert "10:23 [other] 会議の予定について話しました" in result

    def test_handles_fts_syntax_characters(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_mic", started_at_utc="2026-09-15T00:00:00Z", sha256="b" * 64
        )
        _insert_utterance(
            conn,
            recording_id=rec,
            abs_start_utc="2026-09-15T02:00:00Z",
            speaker="me",
            text='今日は"重要な会議"でした',
        )

        # A raw quote is FTS5 query syntax; the tool must treat it as literal text
        # instead of raising a query-language error. (The trigram tokenizer needs at
        # least 3 characters to index anything, hence the longer phrase here.)
        result = mcp_server.search_transcript(query='"重要な会議"')
        assert "[me]" in result
        assert "重要な会議" in result

        # A hyphen is also FTS5 syntax (NOT); this must not raise even with no match.
        result = mcp_server.search_transcript(query="foo-bar-not-present")
        assert "No matches" in result

    def test_no_matches_message(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.search_transcript(query="存在しない単語です")
        assert "No matches" in result


class TestListDays:
    def test_buckets_by_local_date_not_utc(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_mic", started_at_utc="2026-09-15T00:00:00Z", sha256="c" * 64
        )
        # Default timezone is Asia/Tokyo (UTC+9): 20:00Z on the 15th is 05:00 JST on the
        # 16th, while 10:00Z on the 15th is 19:00 JST still on the 15th. Both utterances
        # share a UTC calendar date but must land on different local days.
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc="2026-09-15T20:00:00Z", text="夜遅くの発話"
        )
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc="2026-09-15T10:00:00Z", text="夕方の発話"
        )

        result = mcp_server.list_days()
        assert "2026-09-16: 1 utterance(s)" in result
        assert "2026-09-15: 1 utterance(s)" in result
        # Most recent local day first.
        assert result.index("2026-09-16") < result.index("2026-09-15")

    def test_no_utterances(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.list_days()
        assert "No utterances" in result


class TestVocabulary:
    def test_add_merges_into_existing_and_list_shows_it(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        result = mcp_server.add_vocabulary(
            term="プロジェクトX", aliases=["PX"], note="社内プロジェクト"
        )
        assert "プロジェクトX" in result

        shown = mcp_server.list_vocabulary()
        assert "プロジェクトX" in shown
        assert "PX" in shown

        # Adding the alias as a new term should merge into the same entry, not create
        # a second one.
        mcp_server.add_vocabulary(term="PX", note="")
        shown_again = mcp_server.list_vocabulary()
        assert shown_again.count("プロジェクトX") == 1

    def test_list_empty(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.list_vocabulary()
        assert "empty" in result


class TestToolSafeWrapper:
    def test_runtime_error_becomes_returned_string(self) -> None:
        @mcp_server._tool_safe
        def boom() -> str:
            raise RuntimeError("no API key configured")

        assert boom() == "no API key configured"

    def test_other_exception_is_labelled(self) -> None:
        @mcp_server._tool_safe
        def boom() -> str:
            raise ValueError("bad input")

        assert boom() == "ValueError: bad input"

    def test_status_survives_a_broken_config(
        self, vas: tuple[Config, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail() -> None:
            raise RuntimeError("config is broken")

        monkeypatch.setattr(mcp_server, "status", mcp_server._tool_safe(fail))
        assert mcp_server.status() == "config is broken"


class TestUpdateApp:
    def test_refuses_outside_git_work_tree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        not_a_repo = tmp_path / "not-a-repo"
        not_a_repo.mkdir()
        monkeypatch.setattr(mcp_server, "_repo_dir", lambda: not_a_repo)

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("subprocess must not run outside a git work tree")

        monkeypatch.setattr(mcp_server.subprocess, "run", _fail_if_called)

        result = mcp_server.update_app()
        assert "not a git work tree" in result
        assert str(not_a_repo) in result


def _insert_pending_recording(conn: sqlite3.Connection, *, sha256: str) -> int:
    """A recording row that is pending (no `processed_at`, no `error`)."""
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at, error
        ) VALUES ('mac_mic', 'dev1', '2026-09-15T00:00:00Z', '+00:00', 1000,
                  ?, 'store/x.wav', 'x.wav', '2026-09-15T00:00:00Z', NULL, NULL)
        """,
        (sha256,),
    )
    conn.commit()
    return cur.lastrowid


def _insert_failed_recording(conn: sqlite3.Connection, *, sha256: str) -> int:
    """A recording row that previously failed to transcribe."""
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at, error
        ) VALUES ('mac_mic', 'dev1', '2026-09-15T00:00:00Z', '+00:00', 1000,
                  ?, 'store/x.wav', 'x.wav', '2026-09-15T00:00:00Z', NULL, 'boom')
        """,
        (sha256,),
    )
    conn.commit()
    return cur.lastrowid


class TestRetryFailed:
    def test_does_not_run_asr(
        self, vas: tuple[Config, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an earlier version of this tool ran unbounded ASR inline via
        `pipeline.process_pending`, which could take minutes and blow past Claude
        Desktop's ~60s tool-call timeout. It must now only re-queue."""
        _cfg, conn = vas
        _insert_failed_recording(conn, sha256="f" * 64)

        def _fail_if_called(*args: object, **kwargs: object) -> int:
            raise AssertionError("retry_failed must not run ASR via pipeline.process_pending")

        monkeypatch.setattr("voice_ai_summary.pipeline.process_pending", _fail_if_called)

        result = mcp_server.retry_failed()
        assert "cleared the error flag on 1 recording(s)" in result
        assert "1 recording(s) now pending" in result
        assert "process_pending" in result

    def test_no_failed_recordings(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.retry_failed()
        assert result == "no failed recordings"


class TestProcessPending:
    def test_reports_processed_and_remaining_counts(
        self, vas: tuple[Config, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _cfg, conn = vas
        for i in range(3):
            _insert_pending_recording(conn, sha256=f"{i}" * 64)

        def fake_process_pending(
            conn: sqlite3.Connection, cfg: Config, backend: object, limit: int | None = None
        ) -> int:
            sql = "SELECT id FROM recordings WHERE processed_at IS NULL ORDER BY id"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            ids = [r["id"] for r in conn.execute(sql).fetchall()]
            for rec_id in ids:
                conn.execute(
                    "UPDATE recordings SET processed_at = '2026-09-15T00:00:00Z' WHERE id = ?",
                    (rec_id,),
                )
            conn.commit()
            return len(ids)

        monkeypatch.setattr("voice_ai_summary.pipeline.process_pending", fake_process_pending)

        result = mcp_server.process_pending(limit=2)
        assert "processed 2 recording(s); 1 still pending." in result
        assert "process_pending again" in result

        result2 = mcp_server.process_pending(limit=2)
        assert "processed 1 recording(s); 0 still pending." in result2
        assert "process_pending again" not in result2


class TestThroughput:
    def test_computes_realtime_factor(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn,
            source="mac_system",
            started_at_utc="2026-09-15T01:00:00Z",
            sha256="a" * 64,
            duration_ms=60_000,
            processing_ms=30_000,  # 60s audio / 30s processing -> 2.0x faster than realtime
        )
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc="2026-09-15T01:00:00Z", asr_model="mlx"
        )

        result = mcp_server.throughput()
        assert "audio=60.0s" in result
        assert "processing=30.0s" in result
        assert "2.00x faster than realtime" in result
        assert "model=mlx" in result
        assert "verdict" in result

    def test_groups_by_model_when_sample_has_two(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        _cfg, conn = vas
        cpu_rec = _insert_recording(
            conn,
            source="mac_system",
            started_at_utc="2026-09-15T01:00:00Z",
            sha256="a" * 64,
            duration_ms=60_000,
            processing_ms=75_000,  # slower than realtime
        )
        _insert_utterance(
            conn,
            recording_id=cpu_rec,
            abs_start_utc="2026-09-15T01:00:00Z",
            asr_model="faster-whisper",
        )
        mlx_rec = _insert_recording(
            conn,
            source="mac_system",
            started_at_utc="2026-09-15T02:00:00Z",
            sha256="b" * 64,
            duration_ms=60_000,
            processing_ms=20_000,  # faster than realtime
        )
        _insert_utterance(
            conn, recording_id=mlx_rec, abs_start_utc="2026-09-15T02:00:00Z", asr_model="mlx"
        )

        result = mcp_server.throughput()
        assert "faster-whisper: 1 recording(s)" in result
        assert "mlx: 1 recording(s)" in result
        assert "slower than realtime" in result  # faster-whisper line
        assert "faster than realtime" in result  # mlx line and/or per-recording line
        assert "overall (2 recording(s)" in result

    def test_no_timing_data_says_so(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.throughput()
        assert "No timing data yet" in result


class TestStatusBacklog:
    def test_reports_no_timing_data(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        _insert_recording(
            conn,
            source="mac_mic",
            started_at_utc="2026-09-15T01:00:00Z",
            sha256="a" * 64,
            processed_at=None,
        )

        result = mcp_server.status()
        assert "no timing data yet" in result

    def test_reports_projected_backlog_time(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        _insert_recording(
            conn,
            source="mac_mic",
            started_at_utc="2026-09-15T01:00:00Z",
            sha256="a" * 64,
            duration_ms=600_000,
            processing_ms=300_000,
            processed_at="2026-09-15T01:10:00Z",
        )
        _insert_recording(
            conn,
            source="mac_mic",
            started_at_utc="2026-09-15T02:00:00Z",
            sha256="b" * 64,
            processed_at=None,
        )

        result = mcp_server.status()
        assert "1 recording(s) pending" in result
        assert "2.00x realtime factor" in result


class TestRecent:
    def test_returns_only_utterances_inside_window_in_order_and_reports_lag(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        from datetime import UTC, datetime, timedelta

        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_mic", started_at_utc="2026-09-15T01:00:00Z", sha256="a" * 64
        )
        now = datetime.now(UTC)

        def _iso(dt) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        outside_old = now - timedelta(minutes=90)
        inside_early = now - timedelta(minutes=20)
        inside_late = now - timedelta(minutes=5)

        _insert_utterance(conn, recording_id=rec, abs_start_utc=_iso(outside_old), text="古い発話")
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc=_iso(inside_late), text="最新の発話"
        )
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc=_iso(inside_early), text="少し前の発話"
        )

        result = mcp_server.recent(minutes=30)
        assert "古い発話" not in result
        assert "少し前の発話" in result
        assert "最新の発話" in result
        # Oldest first.
        assert result.index("少し前の発話") < result.index("最新の発話")
        # The header reports how far behind the newest utterance in the DB is right
        # now: the newest inserted utterance is ~5 min old.
        import re

        match = re.search(r"newest utterance in the database is (\d+\.\d+) min old", result)
        assert match is not None
        assert 4.9 <= float(match.group(1)) <= 5.1

    def test_no_utterances_at_all(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.recent()
        assert "no utterances recorded yet" in result
        assert "No utterances in the last" in result


class TestListRecordings:
    def test_lists_two_recordings_for_a_day_sorted_by_time(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        _cfg, conn = vas
        rec_a = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="a" * 64
        )
        rec_b = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:05Z", sha256="b" * 64
        )
        _insert_utterance(
            conn, recording_id=rec_a, abs_start_utc="2026-09-15T01:00:00Z", speaker="other"
        )
        _insert_utterance(
            conn, recording_id=rec_b, abs_start_utc="2026-09-15T01:00:05Z", speaker="other"
        )

        result = mcp_server.list_recordings(day="2026-09-15")
        assert f"id={rec_a}" in result
        assert f"id={rec_b}" in result
        assert result.index(f"id={rec_a}") < result.index(f"id={rec_b}")
        assert "utterances=1" in result
        assert "sha256=aaaaaaaa" in result
        assert "sha256=bbbbbbbb" in result
        assert "path=store/x.wav" in result
        assert "state=processed" in result

    def test_no_recordings_message(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.list_recordings(day="2026-01-01")
        assert "No recordings" in result


class TestFindDuplicates:
    def test_groups_by_text_and_time_proximity_and_names_the_pair(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        _cfg, conn = vas
        rec_a = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="a" * 64
        )
        rec_b = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="b" * 64
        )

        # Mechanism (a): the same audio, transcribed under two different recording ids,
        # close together in time (a `.part` file and its completed re-ingest).
        _insert_utterance(
            conn,
            recording_id=rec_a,
            abs_start_utc="2026-09-15T01:00:00Z",
            speaker="other",
            text="こんにちは、テストです",
        )
        _insert_utterance(
            conn,
            recording_id=rec_b,
            abs_start_utc="2026-09-15T01:00:30Z",
            speaker="other",
            text="こんにちは、テストです",
        )

        # Same text, hours apart on the same local day: a genuinely repeated phrase,
        # not a duplicate - must NOT be grouped.
        _insert_utterance(
            conn,
            recording_id=rec_a,
            abs_start_utc="2026-09-15T02:00:00Z",
            speaker="other",
            text="よろしくお願いします",
        )
        _insert_utterance(
            conn,
            recording_id=rec_a,
            abs_start_utc="2026-09-15T05:00:00Z",
            speaker="other",
            text="よろしくお願いします",
        )

        result = mcp_server.find_duplicates(day="2026-09-15")
        assert "こんにちは、テストです" in result
        assert "よろしくお願いします" not in result
        assert "2 utterance(s) in 1 duplicate group(s)" in result
        assert f"({rec_a}, {rec_b})" in result
        assert f"recording={rec_a}" in result
        assert f"recording={rec_b}" in result

    def test_no_utterances_message(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.find_duplicates(day="2026-01-01")
        assert "No utterances" in result

    def test_no_duplicates_message(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="c" * 64
        )
        _insert_utterance(
            conn, recording_id=rec, abs_start_utc="2026-09-15T01:00:00Z", text="固有の発話"
        )

        result = mcp_server.find_duplicates(day="2026-09-15")
        assert "No duplicate utterances" in result


class TestDropRecording:
    def test_refuses_without_confirm_and_changes_nothing(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        _cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="e" * 64
        )
        _insert_utterance(conn, recording_id=rec, abs_start_utc="2026-09-15T01:00:00Z")

        result = mcp_server.drop_recording(recording_id=rec)
        assert "confirm=True" in result
        assert "mac_system" in result
        assert "utterances=1" in result
        assert "store/x.wav" in result

        assert (
            conn.execute("SELECT COUNT(*) AS n FROM recordings WHERE id = ?", (rec,)).fetchone()[
                "n"
            ]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM utterances WHERE recording_id = ?", (rec,)
            ).fetchone()["n"]
            == 1
        )

    def test_confirm_deletes_rows_but_leaves_file_on_disk(
        self, vas: tuple[Config, sqlite3.Connection]
    ) -> None:
        cfg, conn = vas
        rec = _insert_recording(
            conn, source="mac_system", started_at_utc="2026-09-15T01:00:00Z", sha256="f" * 64
        )
        _insert_utterance(conn, recording_id=rec, abs_start_utc="2026-09-15T01:00:00Z")

        # storage_path in the helper is the literal string "store/x.wav".
        audio_path = cfg.paths.store / "store" / "x.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake audio")

        result = mcp_server.drop_recording(recording_id=rec, confirm=True)
        assert "deleted" in result
        assert "audio file left on disk" in result

        assert (
            conn.execute("SELECT COUNT(*) AS n FROM recordings WHERE id = ?", (rec,)).fetchone()[
                "n"
            ]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM utterances WHERE recording_id = ?", (rec,)
            ).fetchone()["n"]
            == 0
        )
        assert audio_path.exists()
        assert audio_path.read_bytes() == b"fake audio"

    def test_unknown_recording_id(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.drop_recording(recording_id=999999, confirm=True)
        assert "no such recording" in result


class TestWorkerStatus:
    def test_non_macos_says_so_plainly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_server.sys, "platform", "linux")
        assert "macOS-only" in mcp_server.worker_status()

    def test_missing_plist_reports_not_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def fake_run(cmd: list[str], **kwargs: object):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "Could not find service"
            return result

        monkeypatch.setattr(mcp_server.subprocess, "run", fake_run)

        result = mcp_server.worker_status()
        assert "MISSING" in result
        assert "not loaded" in result

    def test_loaded_service_reports_state_and_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from voice_ai_summary import launchd

        monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        agents_dir = tmp_path / "Library" / "LaunchAgents"
        agents_dir.mkdir(parents=True)
        (agents_dir / f"{launchd.WORKER_LABEL}.plist").write_text("<plist/>")

        print_output = (
            f"{launchd.WORKER_LABEL} = {{\n"
            "\tactive count = 1\n"
            "\tstate = running\n"
            "\tpid = 4321\n"
            "\tlast exit status = 0\n"
            "}\n"
        )

        def fake_run(cmd: list[str], **kwargs: object):
            result = MagicMock()
            if cmd[:2] == ["launchctl", "print"] and launchd.WORKER_LABEL in cmd[2]:
                result.returncode = 0
                result.stdout = print_output
                result.stderr = ""
            else:
                result.returncode = 1
                result.stdout = ""
                result.stderr = ""
            return result

        monkeypatch.setattr(mcp_server.subprocess, "run", fake_run)

        result = mcp_server.worker_status()
        assert "state = running" in result
        assert "pid = 4321" in result
        assert "MISSING" in result  # the digest plist was never written in this test


class TestRestartWorker:
    def test_non_macos_says_so_plainly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_server.sys, "platform", "linux")
        assert "macOS-only" in mcp_server.restart_worker()

    def test_refuses_without_plist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("must not shell out to launchctl with no plist installed")

        monkeypatch.setattr(mcp_server.subprocess, "run", _fail_if_called)

        result = mcp_server.restart_worker()
        assert "install_services" in result


class TestServiceLogs:
    def test_rejects_unknown_service_name(self, vas: tuple[Config, sqlite3.Connection]) -> None:
        result = mcp_server.service_logs(service="bogus")
        assert "unknown service" in result

    def test_missing_log_files_say_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = mcp_server.service_logs(service="worker")
        assert "does not exist" in result

    def test_tails_existing_logs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from voice_ai_summary import launchd

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        directory = launchd.log_dir()
        directory.mkdir(parents=True)
        (directory / f"{launchd.WORKER_LABEL}.log").write_text(
            "\n".join(f"line{i}" for i in range(50))
        )
        (directory / f"{launchd.WORKER_LABEL}.err").write_text("")

        result = mcp_server.service_logs(service="worker", lines=5)
        assert "line49" in result
        assert "line45" in result
        assert "line44" not in result
        assert "(empty)" in result


class TestBackendCache:
    """Claude Desktop keeps the server alive between calls, so the ASR model should be
    loaded once, not once per `process_pending` call."""

    def test_backend_is_reused_across_calls(self, vas, monkeypatch) -> None:
        cfg, _conn = vas
        builds = []

        def _fake_get_backend(config):
            builds.append(config.asr.resolved_model)
            return FakeBackend()

        monkeypatch.setattr("voice_ai_summary.asr.get_backend", _fake_get_backend)
        mcp_server._backend_cache.clear()

        first = mcp_server._cached_backend(cfg)
        second = mcp_server._cached_backend(cfg)

        assert first is second
        assert len(builds) == 1

    def test_changing_the_model_rebuilds_it(self, vas, monkeypatch) -> None:
        cfg, _conn = vas
        monkeypatch.setattr("voice_ai_summary.asr.get_backend", lambda config: FakeBackend())
        mcp_server._backend_cache.clear()

        first = mcp_server._cached_backend(cfg)
        cfg.asr.model = "some/other-model"
        second = mcp_server._cached_backend(cfg)

        assert first is not second
        assert len(mcp_server._backend_cache) == 1, "the stale entry should not be kept"


def _wait_for_job(job: mcp_server.Job, timeout: float = 2.0) -> None:
    """Poll until `job` leaves the "running" state, asserting rather than hanging the
    suite if a test's fake worker never finishes."""
    deadline = time.monotonic() + timeout
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert job.state != "running", f"job {job.id} did not finish within {timeout}s"


class TestJobRegistry:
    """Drives the background job registry (`_start_job`/`job_status`) directly with
    fake worker functions - no real Claude API call, no sleep longer than a few
    milliseconds, and every worker thread is joined via `_wait_for_job` with a timeout
    so a bug here can never hang the suite."""

    @pytest.fixture(autouse=True)
    def _clear_jobs(self):
        mcp_server._jobs.clear()
        yield
        mcp_server._jobs.clear()

    def test_job_runs_to_completion_and_status_reports_result(self) -> None:
        def worker(job: mcp_server.Job) -> str:
            return "# digest markdown\n"

        job, started = mcp_server._start_job("test", "day-a", worker)
        assert started is True
        _wait_for_job(job)

        result = mcp_server.job_status(job_id=job.id)
        assert "done" in result
        assert "digest markdown" in result

    def test_failing_job_is_reported_as_failed_and_server_survives(self) -> None:
        def worker(job: mcp_server.Job) -> str:
            raise ValueError("boom")

        job, _started = mcp_server._start_job("test", "day-b", worker)
        _wait_for_job(job)

        result = mcp_server.job_status(job_id=job.id)
        assert "failed" in result
        assert "boom" in result

        # The exception must not have killed the thread silently or taken the
        # registry down - it still answers normally afterwards.
        assert mcp_server.job_status() is not None

    def test_second_start_for_same_key_returns_first_job_without_starting_a_second(
        self,
    ) -> None:
        release = threading.Event()
        calls: list[int] = []

        def worker(job: mcp_server.Job) -> str:
            calls.append(1)
            release.wait(timeout=5)
            return "done"

        job1, started1 = mcp_server._start_job("rebuild_day", "2026-09-16", worker)
        job2, started2 = mcp_server._start_job("rebuild_day", "2026-09-16", worker)

        assert started1 is True
        assert started2 is False
        assert job2 is job1

        release.set()
        _wait_for_job(job1)
        assert len(calls) == 1, "starting the second job must not have run the worker"

    def test_job_status_unknown_id_says_so(self) -> None:
        result = mcp_server.job_status(job_id="does-not-exist")
        assert "No job" in result

    def test_job_status_with_no_id_lists_recent_jobs_newest_first(self) -> None:
        def worker(job: mcp_server.Job) -> str:
            return "ok"

        job_old, _ = mcp_server._start_job("test", "old", worker)
        _wait_for_job(job_old)
        time.sleep(0.01)
        job_new, _ = mcp_server._start_job("test", "new", worker)
        _wait_for_job(job_new)

        result = mcp_server.job_status()
        assert result.index(job_new.id) < result.index(job_old.id)


class TestRebuildDayAsync:
    """`rebuild_day` itself, wired through the background job registry."""

    @pytest.fixture(autouse=True)
    def _clear_jobs(self):
        mcp_server._jobs.clear()
        yield
        mcp_server._jobs.clear()

    def test_returns_promptly_before_the_worker_finishes(
        self, vas: tuple[Config, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: this must not go back to running the rebuild inline - that is
        exactly what let a Claude Desktop tool call blow through its ~60s timeout
        while the paid Claude API work kept going invisibly."""
        release = threading.Event()

        def _blocked_run_day(conn, cfg, day, *, client=None, force=False):
            release.wait(timeout=5)
            return "# blocked digest\n"

        monkeypatch.setattr("voice_ai_summary.summarize.run_day", _blocked_run_day)

        start = time.monotonic()
        result = mcp_server.rebuild_day(day="2026-09-16")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, "rebuild_day must return immediately, not wait on the worker"
        assert "2026-09-16" in result
        assert "Started job" in result

        job = next(
            j
            for j in mcp_server._jobs.values()
            if j.kind == "rebuild_day" and j.key == "2026-09-16"
        )
        release.set()
        _wait_for_job(job)

        status = mcp_server.job_status(job_id=job.id)
        assert "done" in status
        assert "blocked digest" in status

    def test_second_call_for_a_running_day_does_not_start_a_second_job(
        self, vas: tuple[Config, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()
        calls: list[int] = []

        def _blocked_run_day(conn, cfg, day, *, client=None, force=False):
            calls.append(1)
            release.wait(timeout=5)
            return "digest"

        monkeypatch.setattr("voice_ai_summary.summarize.run_day", _blocked_run_day)

        first = mcp_server.rebuild_day(day="2026-09-16")
        second = mcp_server.rebuild_day(day="2026-09-16")

        assert "already running" in second

        release.set()
        job = next(
            j
            for j in mcp_server._jobs.values()
            if j.kind == "rebuild_day" and j.key == "2026-09-16"
        )
        _wait_for_job(job)
        assert len(calls) == 1, "a second tool call must not start a second paid job"
        assert "Started job" in first
