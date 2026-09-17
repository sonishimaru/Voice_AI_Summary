"""Tests for retention: pruning old audio, log lines and oversized launchd logs."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

from voice_ai_summary.config import Config
from voice_ai_summary.db import utcnow_iso
from voice_ai_summary.retention import (
    EVENTS_FILENAME,
    WORKER_LOG_BASENAMES,
    PruneReport,
    prune,
)
from voice_ai_summary.security import AUDIT_FILENAME, open_private


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert_recording(
    conn,
    *,
    sha256: str,
    storage_path: str = "x.wav",
    processed_at: str | None = None,
    error: str | None = None,
    ingested_at: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO recordings(source, started_at_utc, sha256, storage_path, ingested_at,"
        " processed_at, error) VALUES ('mac_mic', '2026-09-01T00:00:00Z', ?, ?, ?, ?, ?)",
        (sha256, storage_path, ingested_at or utcnow_iso(), processed_at, error),
    )
    conn.commit()
    return cur.lastrowid


def _make_audio(cfg: Config, rel: str, data: bytes = b"fake audio bytes") -> Path:
    path = cfg.paths.store / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class TestAudioPruning:
    def test_old_processed_audio_is_deleted_and_stamped(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old_processed = _fmt(now - timedelta(days=8))
        rec_id = _insert_recording(
            conn, sha256="a" * 64, storage_path="a.wav", processed_at=old_processed
        )
        audio = _make_audio(cfg, "a.wav", data=b"0123456789")

        report = prune(conn, cfg, now=now, dry_run=False)

        assert not audio.exists()
        assert report.audio_deleted == 1
        assert report.audio_bytes == 10
        row = conn.execute(
            "SELECT audio_deleted_at FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()
        assert row["audio_deleted_at"] is not None

    def test_recently_processed_audio_is_kept(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        recent_processed = _fmt(now - timedelta(days=6))
        rec_id = _insert_recording(
            conn, sha256="b" * 64, storage_path="b.wav", processed_at=recent_processed
        )
        audio = _make_audio(cfg, "b.wav")

        report = prune(conn, cfg, now=now, dry_run=False)

        assert audio.exists()
        assert report.audio_deleted == 0
        row = conn.execute(
            "SELECT audio_deleted_at FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()
        assert row["audio_deleted_at"] is None

    def test_errored_audio_kept_until_its_own_longer_deadline(
        self, vas: tuple[Config, object]
    ) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        # Old enough for the (shorter) `audio_days` rule but not yet for the (longer)
        # `errored_audio_days` rule - an errored recording must use its own deadline.
        ingested = _fmt(now - timedelta(days=10))
        rec_id = _insert_recording(
            conn, sha256="c" * 64, storage_path="c.wav", error="boom", ingested_at=ingested
        )
        audio = _make_audio(cfg, "c.wav")

        report = prune(conn, cfg, now=now, dry_run=False)
        assert audio.exists()
        assert report.audio_deleted == 0

        # Now old enough for errored_audio_days (30) too.
        very_old = _fmt(now - timedelta(days=31))
        conn.execute("UPDATE recordings SET ingested_at = ? WHERE id = ?", (very_old, rec_id))
        conn.commit()

        report2 = prune(conn, cfg, now=now, dry_run=False)
        assert not audio.exists()
        assert report2.audio_deleted == 1

    def test_pending_audio_is_never_deleted(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        # Neither processed nor errored: pending, however old `ingested_at` is.
        very_old = _fmt(now - timedelta(days=1000))
        rec_id = _insert_recording(
            conn, sha256="d" * 64, storage_path="d.wav", ingested_at=very_old
        )
        audio = _make_audio(cfg, "d.wav")

        report = prune(conn, cfg, now=now, dry_run=False)

        assert audio.exists()
        assert report.audio_deleted == 0
        row = conn.execute(
            "SELECT audio_deleted_at FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()
        assert row["audio_deleted_at"] is None

    def test_audio_days_zero_disables_the_rule(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        cfg.retention.audio_days = 0
        now = datetime.now(UTC)
        old_processed = _fmt(now - timedelta(days=365))
        _insert_recording(conn, sha256="e" * 64, storage_path="e.wav", processed_at=old_processed)
        audio = _make_audio(cfg, "e.wav")

        report = prune(conn, cfg, now=now, dry_run=False)

        assert audio.exists()
        assert report.audio_deleted == 0

    def test_missing_file_is_still_stamped_and_counted(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old_processed = _fmt(now - timedelta(days=8))
        rec_id = _insert_recording(
            conn, sha256="f" * 64, storage_path="missing.wav", processed_at=old_processed
        )
        # Deliberately do not create the file: it's already gone (e.g. deleted by hand).

        report = prune(conn, cfg, now=now, dry_run=False)

        assert report.audio_missing == 1
        assert report.audio_deleted == 0
        row = conn.execute(
            "SELECT audio_deleted_at FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()
        assert row["audio_deleted_at"] is not None

    def test_dry_run_touches_neither_disk_nor_db(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old_processed = _fmt(now - timedelta(days=8))
        rec_id = _insert_recording(
            conn, sha256="g" * 64, storage_path="g.wav", processed_at=old_processed
        )
        audio = _make_audio(cfg, "g.wav", data=b"0123456789")

        report = prune(conn, cfg, now=now, dry_run=True)

        assert audio.exists()
        assert report.dry_run is True
        assert report.audio_deleted == 1  # counted for the preview...
        assert report.audio_bytes == 10
        row = conn.execute(
            "SELECT audio_deleted_at FROM recordings WHERE id = ?", (rec_id,)
        ).fetchone()
        assert row["audio_deleted_at"] is None  # ...but nothing was actually stamped


class TestJsonlPruning:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open_private(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_drops_old_lines_keeps_new_and_unparsable(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old = _fmt(now - timedelta(days=400))
        new = _fmt(now - timedelta(days=1))
        usage_path = cfg.paths.root / "usage.jsonl"
        self._write_jsonl(
            usage_path,
            [
                {"at": old, "purpose": "map", "model": "m", "input": 1, "output": 1},
                {"at": new, "purpose": "map", "model": "m", "input": 2, "output": 2},
            ],
        )
        # An unparsable line - garbage JSON - must survive pruning untouched.
        with open_private(usage_path, "a") as f:
            f.write("not json at all\n")
        # A group of files created directly by the test, not through `open_private`,
        # should not already be 0600 - otherwise the "result is 0600" assertion below
        # would be trivially true even if the rewrite path were broken.
        usage_path.chmod(0o644)

        report = prune(conn, cfg, now=now, dry_run=False)

        assert report.usage_lines_dropped == 1
        lines = usage_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert new in lines[0]
        assert lines[1] == "not json at all"
        assert stat.S_IMODE(usage_path.stat().st_mode) == 0o600

    def test_recorder_events_and_audit_logs_are_pruned_too(
        self, vas: tuple[Config, object]
    ) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old = _fmt(now - timedelta(days=500))
        new = _fmt(now - timedelta(days=1))

        events_path = cfg.paths.root / EVENTS_FILENAME
        self._write_jsonl(events_path, [{"at": old, "kind": "x"}, {"at": new, "kind": "y"}])

        audit_path = cfg.paths.root / AUDIT_FILENAME
        self._write_jsonl(
            audit_path,
            [
                {"ts": old, "tool": "t", "args": {}, "outcome": "ok"},
                {"ts": new, "tool": "t", "args": {}, "outcome": "ok"},
            ],
        )

        report = prune(conn, cfg, now=now, dry_run=False)

        assert report.events_lines_dropped == 1
        assert report.audit_lines_dropped == 1
        assert len(events_path.read_text(encoding="utf-8").splitlines()) == 1
        assert len(audit_path.read_text(encoding="utf-8").splitlines()) == 1

    def test_days_zero_disables_jsonl_pruning(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        cfg.retention.usage_log_days = 0
        now = datetime.now(UTC)
        old = _fmt(now - timedelta(days=10000))
        usage_path = cfg.paths.root / "usage.jsonl"
        self._write_jsonl(usage_path, [{"at": old, "purpose": "map"}])

        report = prune(conn, cfg, now=now, dry_run=False)

        assert report.usage_lines_dropped == 0
        assert len(usage_path.read_text(encoding="utf-8").splitlines()) == 1


class TestLaunchdLogTruncation:
    def test_truncates_oversized_log_to_a_line_aligned_tail(
        self, vas: tuple[Config, object]
    ) -> None:
        cfg, conn = vas
        cfg.retention.log_max_bytes = 100
        log_dir = cfg.paths.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "com.voiceaisummary.worker.log"
        lines = [f"line {i:03d} - padding to make this long enough" for i in range(20)]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        original_size = log_path.stat().st_size
        assert original_size > 100

        report = prune(conn, cfg, dry_run=False, log_dir=log_dir)

        assert log_path.name in report.logs_truncated
        new_size = log_path.stat().st_size
        assert new_size < original_size
        # Aligned to a line boundary: every remaining line is one of the originals,
        # never a fragment of one.
        remaining = log_path.read_text(encoding="utf-8").splitlines()
        assert remaining  # something survived
        for line in remaining:
            assert line in lines
        # The tail (most recent lines) is what's kept, not the head.
        assert remaining[-1] == lines[-1]

    def test_skip_logs_are_left_alone(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        cfg.retention.log_max_bytes = 50
        log_dir = cfg.paths.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        skipped = log_dir / "com.voiceaisummary.worker.log"
        skipped.write_text("x" * 500, encoding="utf-8")

        report = prune(conn, cfg, dry_run=False, log_dir=log_dir, skip_logs=WORKER_LOG_BASENAMES)

        assert skipped.name not in report.logs_truncated
        assert skipped.stat().st_size == 500


class TestEmptyDirRemoval:
    def test_empty_store_directories_are_removed(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        now = datetime.now(UTC)
        old_processed = _fmt(now - timedelta(days=8))
        _insert_recording(
            conn,
            sha256="h" * 64,
            storage_path="2026/09/01/h.wav",
            processed_at=old_processed,
        )
        _make_audio(cfg, "2026/09/01/h.wav")
        # An unrelated day that still has audio: must survive.
        _insert_recording(
            conn,
            sha256="i" * 64,
            storage_path="2026/09/02/i.wav",
            processed_at=_fmt(now - timedelta(days=1)),
        )
        _make_audio(cfg, "2026/09/02/i.wav")

        report = prune(conn, cfg, now=now, dry_run=False)

        assert report.empty_dirs_removed >= 1
        assert not (cfg.paths.store / "2026" / "09" / "01").exists()
        assert (cfg.paths.store / "2026" / "09" / "02" / "i.wav").exists()


def test_prune_report_summary_mentions_the_counts() -> None:
    report = PruneReport(
        audio_deleted=3,
        audio_bytes=12345,
        audio_missing=1,
        audio_skipped_errors=0,
        usage_lines_dropped=2,
        events_lines_dropped=4,
        audit_lines_dropped=0,
        logs_truncated=["com.voiceaisummary.worker.log"],
        empty_dirs_removed=1,
        dry_run=False,
    )
    text = report.summary()
    assert "3" in text
    assert "12345" in text
    assert "1" in text  # audio_missing
    assert "2" in text  # usage_lines_dropped
    assert "4" in text  # events_lines_dropped
    assert "com.voiceaisummary.worker.log" in text


def test_dry_run_summary_says_so() -> None:
    assert "dry run" in PruneReport(dry_run=True).summary().lower()
