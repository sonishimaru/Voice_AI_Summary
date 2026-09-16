"""Tests for the MCP stdio server's tool functions.

Tool functions are called directly (module-level, `@_tool_safe`-wrapped callables) -
no MCP transport, no network, no Anthropic API calls.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voice_ai_summary import mcp_server
from voice_ai_summary.config import Config
from voice_ai_summary.summarize import PROMPT_VERSION


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
) -> int:
    cur = conn.execute(
        """
        INSERT INTO utterances(
            recording_id, t_start_ms, t_end_ms, abs_start_utc, text, lang,
            asr_model, avg_logprob, speaker
        ) VALUES (?, ?, ?, ?, ?, 'ja', 'fake', -0.1, ?)
        """,
        (recording_id, t_start_ms, t_start_ms + 500, abs_start_utc, text, speaker),
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
