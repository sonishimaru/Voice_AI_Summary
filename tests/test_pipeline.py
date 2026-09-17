"""Tests for the VAD→ASR→SQLite pipeline and search."""

from __future__ import annotations

import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from voice_ai_summary import vad as vad_module
from voice_ai_summary.asr import FakeBackend
from voice_ai_summary.config import Config
from voice_ai_summary.db import utcnow_iso
from voice_ai_summary.ingest import ingest_file
from voice_ai_summary.pipeline import (
    delete_range,
    delete_recording,
    process_recording,
    recordings_overlapping,
    speaker_for_source,
)
from voice_ai_summary.search import search
from voice_ai_summary.vad import SpeechRegion


def _write_wav(path: Path, seconds: float = 4.0, sample_rate: int = 16000) -> None:
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)


def test_speaker_for_source() -> None:
    assert speaker_for_source("mac_mic") == "me"
    assert speaker_for_source("mac_system") == "other"
    assert speaker_for_source("file") == "unknown"
    assert speaker_for_source("anything_else") == "unknown"


def test_process_recording_and_search(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)

    fixed_regions = [SpeechRegion(0, 1000, 0.9), SpeechRegion(1500, 2500, 0.8)]
    monkeypatch.setattr(vad_module, "detect_speech", lambda samples, cfg: fixed_regions)

    backend = FakeBackend(["こんにちは、テストです", "今日は会議があります"])
    count = process_recording(conn, cfg, rec_id, backend)
    assert count == 2

    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert row["processed_at"] is not None
    assert row["duration_ms"] is not None

    segments = conn.execute(
        "SELECT * FROM segments WHERE recording_id = ? ORDER BY start_ms", (rec_id,)
    ).fetchall()
    assert len(segments) == 2

    utterances = conn.execute(
        "SELECT * FROM utterances WHERE recording_id = ? ORDER BY t_start_ms", (rec_id,)
    ).fetchall()
    assert [u["text"] for u in utterances] == ["こんにちは、テストです", "今日は会議があります"]
    assert all(u["speaker"] == "other" for u in utterances)
    assert all(u["asr_model"] == "fake" for u in utterances)

    # Second region starts at 1500ms -> abs_start_utc offset from started_at_utc.
    assert utterances[1]["t_start_ms"] == 1500
    assert utterances[1]["abs_start_utc"] == "2026-09-15T00:00:01Z"

    results = search(conn, "会議が")  # 3 chars: exercises the trigram FTS index
    assert len(results) == 1
    assert results[0]["text"] == "今日は会議があります"
    assert results[0]["speaker"] == "other"
    assert results[0]["source"] == "mac_system"

    # Query shorter than 3 chars can't match a trigram index; falls back to LIKE.
    short_results = search(conn, "会議")
    assert len(short_results) == 1
    assert short_results[0]["text"] == "今日は会議があります"


def test_a_second_process_can_write_while_transcription_runs(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoding a recording takes minutes. Holding the write lock across it locks out the
    worker running alongside, which then dies with "database is locked"."""
    from voice_ai_summary.db import connect

    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )

    writes_from_other_process: list[str] = []

    class _WritingBackend(FakeBackend):
        def transcribe(self, samples, *, language, source=None):
            other = connect(cfg.paths.db_path)
            try:
                other.execute("UPDATE recordings SET device_id = 'other' WHERE id = ?", (rec_id,))
                other.commit()
                writes_from_other_process.append("ok")
            finally:
                other.close()
            return super().transcribe(samples, language=language, source=source)

    assert process_recording(conn, cfg, rec_id, _WritingBackend(["文字起こし"])) == 1
    assert writes_from_other_process == ["ok"]


def test_process_recording_records_processing_ms(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`processing_ms` is measured with `time.monotonic()` around VAD + the ASR decode
    loop, and stored alongside `processed_at`."""
    import numpy as np

    import voice_ai_summary.pipeline as pipeline_module

    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)

    # Avoid decoding real audio through the faster-whisper/PyAV stack here: that code
    # path may itself call `time.monotonic()` internally, and this test's fake clock
    # (below) is a global patch that must only see the two calls `process_recording`
    # itself makes around VAD + the ASR loop.
    monkeypatch.setattr(
        pipeline_module, "load_audio_16k", lambda path: np.zeros(16000, dtype="float32")
    )
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )

    # Fake clock: the first call pipeline.py makes (decode_start) returns 100.0, the
    # second (right after the ASR loop) returns 107.5 -> elapsed 7.5s -> 7500ms,
    # regardless of real wall time.
    clock_values = [100.0, 107.5]

    def _fake_monotonic() -> float:
        return clock_values.pop(0) if clock_values else 107.5

    monkeypatch.setattr(pipeline_module.time, "monotonic", _fake_monotonic)

    backend = FakeBackend(["テスト"])
    assert process_recording(conn, cfg, rec_id, backend) == 1

    row = conn.execute(
        "SELECT processing_ms, processed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["processing_ms"] == 7500
    assert row["processed_at"] is not None


def test_process_recording_skips_empty_utterances(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)

    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )
    backend = FakeBackend(["   "])
    count = process_recording(conn, cfg, rec_id, backend)
    assert count == 0


def test_process_recording_sets_error_on_failure(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    conn.execute(
        "UPDATE recordings SET storage_path = 'does/not/exist.wav' WHERE id = ?", (rec_id,)
    )
    conn.commit()

    # Simulate a live claim on the row, as `process_pending` would leave one while this
    # call is decoding it - a failure must clear it just like a success does.
    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (utcnow_iso(), rec_id))
    conn.commit()

    backend = FakeBackend(["x"])
    with pytest.raises(FileNotFoundError):
        process_recording(conn, cfg, rec_id, backend)

    row = conn.execute(
        "SELECT error, claimed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"]
    assert row["claimed_at"] is None


def test_retry_failed_renames_part_and_reprocesses(vas, monkeypatch) -> None:
    from voice_ai_summary.pipeline import process_pending, retry_failed

    cfg, conn = vas
    wav = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(wav, seconds=2)
    rec_id = ingest_file(conn, cfg, wav)
    # Simulate an ingest that grabbed the recorder's still-open file.
    stored = (
        cfg.paths.store
        / conn.execute("SELECT storage_path FROM recordings WHERE id = ?", (rec_id,)).fetchone()[
            "storage_path"
        ]
    )
    part = stored.with_name(stored.name + ".part")
    stored.rename(part)
    conn.execute(
        # claimed_at set too: simulates a worker that had claimed the row and was killed
        # (e.g. `launchctl kickstart -k`) before it got to record the failure.
        "UPDATE recordings SET error = 'boom', claimed_at = ?, storage_path = ? WHERE id = ?",
        (utcnow_iso(), str(part.relative_to(cfg.paths.store)), rec_id),
    )
    conn.commit()

    assert retry_failed(conn, cfg) == [rec_id]
    assert stored.exists() and not part.exists()
    row = conn.execute(
        "SELECT error, processed_at, claimed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is None and row["processed_at"] is None
    # Otherwise the retried recording stays invisible to `process_pending` until the
    # stale-claim timeout expires, even though it is pending again right now.
    assert row["claimed_at"] is None

    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, None)]
    )
    assert process_pending(conn, cfg, FakeBackend(["再処理"])) == 1
    assert conn.execute("SELECT text FROM utterances").fetchone()["text"] == "再処理"


def test_process_recording_is_idempotent(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running `process_recording` on the same recording must replace its
    segments/utterances, not append a second copy - and the FTS mirror must reflect
    exactly the surviving rows, with no stale or doubled entries."""
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)

    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )

    assert process_recording(conn, cfg, rec_id, FakeBackend(["重複しないテスト"])) == 1
    # Re-process the same recording, as would happen if a `.part` file were renamed to
    # its final name and re-queued, or the worker ran on it twice.
    assert process_recording(conn, cfg, rec_id, FakeBackend(["重複しないテスト"])) == 1

    utterances = conn.execute(
        "SELECT * FROM utterances WHERE recording_id = ?", (rec_id,)
    ).fetchall()
    assert len(utterances) == 1
    segments = conn.execute("SELECT * FROM segments WHERE recording_id = ?", (rec_id,)).fetchall()
    assert len(segments) == 1

    results = search(conn, "重複しない")
    assert len(results) == 1
    assert results[0]["text"] == "重複しないテスト"

    # The FTS content table must have exactly as many rows as `utterances` - no leftover
    # rows from the first pass and no doubled row from the second.
    fts_count = conn.execute("SELECT COUNT(*) AS n FROM utterances_fts").fetchone()["n"]
    assert fts_count == 1


def test_reset_recordings_clears_transcript_and_requeues(vas, monkeypatch) -> None:
    from voice_ai_summary.pipeline import process_pending, reset_recordings

    cfg, conn = vas
    wav = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(wav, seconds=2)
    rec_id = ingest_file(conn, cfg, wav)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, None)]
    )
    process_pending(conn, cfg, FakeBackend(["一回目"]))
    # A stale claim left over from some earlier, unrelated run - reset_recordings must
    # clear this too, or the recording stays invisible to process_pending below until
    # the stale-claim timeout expires.
    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (utcnow_iso(), rec_id))
    conn.commit()

    assert reset_recordings(conn, [rec_id]) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"] == 0
    assert (
        conn.execute("SELECT claimed_at FROM recordings WHERE id = ?", (rec_id,)).fetchone()[
            "claimed_at"
        ]
        is None
    )
    assert process_pending(conn, cfg, FakeBackend(["二回目"])) == 1
    assert conn.execute("SELECT text FROM utterances").fetchone()["text"] == "二回目"


def test_process_recording_passes_the_recording_source_to_the_backend(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`process_recording` must thread `row["source"]` through to `backend.transcribe`
    the same way it already threads it into `vad.for_source` - so a per-source ASR
    override (e.g. `[asr.normalize_by_source]`) actually reaches the decoder."""
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )

    received_sources: list[str | None] = []

    class _SourceCapturingBackend(FakeBackend):
        def transcribe(self, samples, *, language, source=None):
            received_sources.append(source)
            return super().transcribe(samples, language=language, source=source)

    cfg.asr.normalize = "none"
    cfg.asr.normalize_by_source = {"mac_mic": "rms"}

    assert process_recording(conn, cfg, rec_id, _SourceCapturingBackend(["テスト"])) == 1

    assert received_sources == ["mac_mic"]
    # And that source resolves to the mic's override, not the global default.
    assert cfg.asr.for_source(received_sources[0]).normalize == "rms"
    assert cfg.asr.for_source("mac_system").normalize == "none"


def test_vad_threshold_can_be_overridden_per_source() -> None:
    """The mic track records far quieter than the system track; one threshold
    over-triggers on the quiet one and under-triggers on the loud one."""
    from voice_ai_summary.config import VadConfig

    cfg = VadConfig(threshold=0.5, threshold_by_source={"mac_mic": 0.7})

    assert cfg.for_source("mac_mic").threshold == 0.7
    assert cfg.for_source("mac_system").threshold == 0.5
    assert cfg.for_source("mac_system") is cfg


def test_search_day_uses_local_dates_and_tolerates_fts_syntax(vas, monkeypatch) -> None:
    """`--day` is a local date, and punctuation is searched literally, not as FTS syntax."""
    from voice_ai_summary.search import fts_query

    cfg, conn = vas
    # 2026-09-14T16:00:00Z is 2026-09-15 01:00 JST.
    rec = conn.execute(
        "INSERT INTO recordings(source, device_id, started_at_utc, tz_offset, sha256,"
        " storage_path, original_name, ingested_at)"
        " VALUES ('mac_mic', 'd', '2026-09-14T16:00:00Z', '+09:00', ?, 'x', 'x', 'x')",
        ("f" * 64,),
    ).lastrowid
    conn.execute(
        "INSERT INTO utterances(recording_id, t_start_ms, t_end_ms, abs_start_utc, text, speaker)"
        " VALUES (?, 0, 500, '2026-09-14T16:00:00Z', 'A-1 の見積もりです', 'me')",
        (rec,),
    )
    conn.commit()

    assert len(search(conn, "見積もり", day="2026-09-15", tz="Asia/Tokyo")) == 1
    assert search(conn, "見積もり", day="2026-09-14", tz="Asia/Tokyo") == []
    assert len(search(conn, "A-1", tz="Asia/Tokyo")) == 1  # would be a syntax error unquoted
    assert search(conn, '"foo', tz="Asia/Tokyo") == []
    assert fts_query('A-1 "b c"') == '"A-1" "b c"'


def test_process_pending_claim_lets_only_one_caller_transcribe(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker and the `process_pending` MCP tool poll the same database and can call
    `process_pending` at the same time. Without an atomic claim, both would `SELECT` the
    same pending id and both spend the (multi-minute) decode on it. Assert on how many
    times `transcribe` actually ran, not just on the resulting rows, so this proves the
    duplicated work is gone rather than merely that the transcript looks right."""
    from voice_ai_summary.db import connect
    from voice_ai_summary.pipeline import process_pending

    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    ingest_file(conn, cfg, src)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )

    calls: list[str] = []

    class _CountingBackend(FakeBackend):
        def __init__(self, tag: str) -> None:
            super().__init__(["テスト"])
            self._tag = tag

        def transcribe(self, samples, *, language, source=None):
            calls.append(self._tag)
            return super().transcribe(samples, language=language, source=source)

    other_conn = connect(cfg.paths.db_path)
    try:

        class _RacingBackend(_CountingBackend):
            """Mid-decode, a second connection (the concurrent poller) tries to claim
            and process the same still-pending recording - exactly the window the
            atomic claim in `process_pending` is meant to close."""

            def transcribe(self, samples, *, language, source=None):
                second_caller_processed = process_pending(
                    other_conn, cfg, _CountingBackend("second")
                )
                assert second_caller_processed == 0
                return super().transcribe(samples, language=language, source=source)

        processed = process_pending(conn, cfg, _RacingBackend("first"))
    finally:
        other_conn.close()

    assert processed == 1
    # "second" never appears: its claim lost the race, so it skipped the recording
    # instead of decoding it a second time.
    assert calls == ["first"]


def test_process_pending_reclaims_a_stale_claim_but_not_a_fresh_one(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim old enough to exceed `CLAIM_TIMEOUT_S` is treated as abandoned (e.g. a
    worker killed mid-decode by `launchctl kickstart -k`) and reclaimed. A claim still
    within the timeout is a live owner's, and `process_pending` must leave it alone."""
    from voice_ai_summary.pipeline import CLAIM_TIMEOUT_S, process_pending

    cfg, conn = vas
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, None)]
    )

    stale_wav = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(stale_wav, seconds=2)
    stale_id = ingest_file(conn, cfg, stale_wav)
    stale_claim = (datetime.now(UTC) - timedelta(seconds=CLAIM_TIMEOUT_S * 2)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (stale_claim, stale_id))

    fresh_wav = cfg.paths.inbox / "mac_mic_dev1_20260915T020304Z.wav"
    _write_wav(fresh_wav, seconds=3)  # different length -> different sha256, distinct row
    fresh_id = ingest_file(conn, cfg, fresh_wav)
    fresh_claim = utcnow_iso()
    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (fresh_claim, fresh_id))
    conn.commit()

    assert process_pending(conn, cfg, FakeBackend(["再開"])) == 1

    stale_row = conn.execute(
        "SELECT processed_at, claimed_at FROM recordings WHERE id = ?", (stale_id,)
    ).fetchone()
    assert stale_row["processed_at"] is not None  # reclaimed and processed
    assert stale_row["claimed_at"] is None  # cleared on success

    fresh_row = conn.execute(
        "SELECT processed_at, claimed_at FROM recordings WHERE id = ?", (fresh_id,)
    ).fetchone()
    assert fresh_row["processed_at"] is None  # left pending: someone else owns this claim
    assert fresh_row["claimed_at"] == fresh_claim  # untouched


def _insert_recording_row(
    conn,
    *,
    sha256: str,
    duration_ms: int | None = None,
    processing_ms: int | None = None,
    processed_at: str | None = None,
    error: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO recordings(source, started_at_utc, duration_ms, sha256, storage_path, "
        "ingested_at, processed_at, error, processing_ms) VALUES "
        "('mac_mic', '2026-09-15T00:00:00Z', ?, ?, 'x.wav', '2026-09-15T00:00:00Z', ?, ?, ?)",
        (duration_ms, sha256, processed_at, error, processing_ms),
    )
    conn.commit()
    return cur.lastrowid


class TestBacklogEta:
    def test_no_pending_recordings(self, vas: tuple[Config, object]) -> None:
        from voice_ai_summary.pipeline import backlog_eta

        _cfg, conn = vas
        assert backlog_eta(conn) == "backlog: none pending."

    def test_no_timing_data_says_so_instead_of_a_number(self, vas: tuple[Config, object]) -> None:
        from voice_ai_summary.pipeline import backlog_eta

        _cfg, conn = vas
        _insert_recording_row(conn, sha256="a" * 64)  # pending, no timing anywhere yet

        result = backlog_eta(conn)
        assert "1 recording(s) pending" in result
        assert "no timing data yet" in result
        # No factor or minute figure must be invented.
        import re

        assert not re.search(r"\d+\.\d+x", result)

    def test_projects_from_recent_factor_and_average_duration(
        self, vas: tuple[Config, object]
    ) -> None:
        from voice_ai_summary.pipeline import backlog_eta

        _cfg, conn = vas
        # Two already-processed recordings: 10 min of audio each, decoded in 5 min each
        # -> average duration 600_000ms, realtime factor 2.0x.
        _insert_recording_row(
            conn,
            sha256="a" * 64,
            duration_ms=600_000,
            processing_ms=300_000,
            processed_at="2026-09-15T00:10:00Z",
        )
        _insert_recording_row(
            conn,
            sha256="b" * 64,
            duration_ms=600_000,
            processing_ms=300_000,
            processed_at="2026-09-15T00:20:00Z",
        )
        # Three pending recordings, each ~10 min of audio once processed.
        for i in range(3):
            _insert_recording_row(conn, sha256=f"{i}pending".ljust(64, "0"))

        result = backlog_eta(conn)
        assert "3 recording(s) pending" in result
        # Estimated audio: 3 * 10 min = 30 min.
        assert "30.0 min" in result
        # At 2.0x realtime, 30 min of audio takes ~15 min to decode.
        assert "2.00x" in result
        assert "15.0 min" in result


# --- retention guards on the pipeline (process_recording / reset_recordings /
# retry_failed / delete_recording) ---------------------------------------------------


def test_process_recording_on_audio_deleted_row_sets_error_and_raises(
    vas: tuple[Config, object],
) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    conn.execute(
        "UPDATE recordings SET audio_deleted_at = '2026-09-20T00:00:00Z' WHERE id = ?",
        (rec_id,),
    )
    conn.commit()

    with pytest.raises(RuntimeError, match="audio was deleted by retention"):
        process_recording(conn, cfg, rec_id, FakeBackend(["x"]))

    row = conn.execute(
        "SELECT error, claimed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is not None
    assert "2026-09-20T00:00:00Z" in row["error"]
    assert row["claimed_at"] is None


def test_reset_recordings_skips_audio_deleted_rows(vas: tuple[Config, object]) -> None:
    from voice_ai_summary.pipeline import reset_recordings

    cfg, conn = vas
    src1 = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    src2 = cfg.paths.inbox / "mac_mic_dev1_20260915T010000Z.wav"
    _write_wav(src1, seconds=2)
    _write_wav(src2, seconds=3)
    ok_id = ingest_file(conn, cfg, src1)
    deleted_id = ingest_file(conn, cfg, src2)
    conn.execute(
        "UPDATE recordings SET audio_deleted_at = '2026-09-20T00:00:00Z' WHERE id = ?",
        (deleted_id,),
    )
    conn.commit()

    assert reset_recordings(conn, [ok_id, deleted_id]) == 1

    ok_row = conn.execute(
        "SELECT processed_at, error FROM recordings WHERE id = ?", (ok_id,)
    ).fetchone()
    assert ok_row["processed_at"] is None and ok_row["error"] is None

    deleted_row = conn.execute(
        "SELECT audio_deleted_at FROM recordings WHERE id = ?", (deleted_id,)
    ).fetchone()
    assert deleted_row["audio_deleted_at"] == "2026-09-20T00:00:00Z"  # untouched


def test_retry_failed_leaves_audio_deleted_rows_errored(vas: tuple[Config, object]) -> None:
    from voice_ai_summary.pipeline import retry_failed

    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    conn.execute(
        "UPDATE recordings SET error = 'audio was deleted by retention on X',"
        " audio_deleted_at = '2026-09-20T00:00:00Z' WHERE id = ?",
        (rec_id,),
    )
    conn.commit()

    assert retry_failed(conn, cfg) == []

    row = conn.execute(
        "SELECT error, processed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is not None
    assert row["processed_at"] is None


def test_delete_recording_with_delete_audio_removes_file(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    storage_path = conn.execute(
        "SELECT storage_path FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()["storage_path"]
    audio_path = cfg.paths.store / storage_path
    assert audio_path.exists()

    assert delete_recording(conn, rec_id, cfg=cfg, delete_audio=True) is True

    assert not audio_path.exists()
    assert conn.execute("SELECT id FROM recordings WHERE id = ?", (rec_id,)).fetchone() is None


def test_delete_recording_without_delete_audio_leaves_file(vas: tuple[Config, object]) -> None:
    """Unchanged default behaviour: existing callers passing only (conn, id) must keep
    working exactly as before - the audio file is left on disk."""
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    storage_path = conn.execute(
        "SELECT storage_path FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()["storage_path"]
    audio_path = cfg.paths.store / storage_path

    assert delete_recording(conn, rec_id) is True
    assert audio_path.exists()


# --- recordings_overlapping / delete_range -------------------------------------------


def _insert_overlap_row(conn, *, sha256: str, started_at_utc: str, duration_ms: int | None) -> int:
    cur = conn.execute(
        "INSERT INTO recordings(source, started_at_utc, duration_ms, sha256, storage_path,"
        " ingested_at) VALUES ('mac_mic', ?, ?, ?, 'x.wav', ?)",
        (started_at_utc, duration_ms, sha256, utcnow_iso()),
    )
    conn.commit()
    return cur.lastrowid


class TestRecordingsOverlapping:
    def test_recording_straddling_the_start_overlaps(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        _insert_overlap_row(
            conn, sha256="a" * 64, started_at_utc="2026-09-15T00:55:00Z", duration_ms=600_000
        )
        rows = recordings_overlapping(
            conn,
            "2026-09-15T01:00:00Z",
            "2026-09-15T01:10:00Z",
            default_duration_ms=900_000,
        )
        assert len(rows) == 1

    def test_recording_ending_exactly_at_start_is_excluded(
        self, vas: tuple[Config, object]
    ) -> None:
        cfg, conn = vas
        # Ends at exactly 2026-09-15T01:00:00Z (started 00:50 + 600_000ms = 10min).
        _insert_overlap_row(
            conn, sha256="b" * 64, started_at_utc="2026-09-15T00:50:00Z", duration_ms=600_000
        )
        rows = recordings_overlapping(
            conn,
            "2026-09-15T01:00:00Z",
            "2026-09-15T01:10:00Z",
            default_duration_ms=900_000,
        )
        assert rows == []

    def test_pending_recording_uses_default_duration(self, vas: tuple[Config, object]) -> None:
        cfg, conn = vas
        # No duration_ms yet (pending) - only overlaps if the *default* duration reaches
        # into the requested range.
        _insert_overlap_row(
            conn, sha256="c" * 64, started_at_utc="2026-09-15T00:50:00Z", duration_ms=None
        )
        # default_duration_ms=600_000 (10min) -> ends at 01:00:00Z, exactly at start: excluded.
        assert (
            recordings_overlapping(
                conn,
                "2026-09-15T01:00:00Z",
                "2026-09-15T01:10:00Z",
                default_duration_ms=600_000,
            )
            == []
        )
        # default_duration_ms=900_000 (15min) -> ends at 01:05:00Z: overlaps.
        rows = recordings_overlapping(
            conn,
            "2026-09-15T01:00:00Z",
            "2026-09-15T01:10:00Z",
            default_duration_ms=900_000,
        )
        assert len(rows) == 1


def _setup_day_for_delete_range(cfg: Config, conn, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build a full day around one recording: ingest + transcribe it, build its episode,
    stash fake episode/day summaries and digest files (+ mirror), and drop a matching
    not-yet-ingested inbox file (plus an unrelated one and a `.part`) - everything
    `delete_range` is expected to touch, or deliberately not touch."""
    from voice_ai_summary.episodes import build_episodes, episode_transcript
    from voice_ai_summary.summarize import PROMPT_VERSION, _content_key

    day = "2026-09-15"
    tz = cfg.summarize.timezone  # Asia/Tokyo by default -> 2026-09-15T01:00:00Z == 10:00 JST
    started_at_utc = "2026-09-15T01:00:00Z"

    src = cfg.paths.inbox / "mac_mic_dev1_20260915T010000Z.wav"
    _write_wav(src, seconds=4)
    rec_id = ingest_file(conn, cfg, src)

    fixed_regions = [SpeechRegion(0, 1000, 0.9)]
    monkeypatch.setattr(vad_module, "detect_speech", lambda samples, cfg: fixed_regions)
    process_recording(conn, cfg, rec_id, FakeBackend(["削除される発言です"]))

    episode_ids = build_episodes(conn, cfg, day)
    assert len(episode_ids) == 1
    episode_id = episode_ids[0]
    transcript = episode_transcript(conn, episode_id, tz)
    episode_key = _content_key(transcript)

    now = utcnow_iso()
    conn.execute(
        "INSERT INTO summaries(scope, scope_key, model, prompt_version, json, markdown,"
        " created_at) VALUES ('episode', ?, 'test', ?, '{}', NULL, ?)",
        (episode_key, PROMPT_VERSION, now),
    )
    conn.execute(
        "INSERT INTO summaries(scope, scope_key, model, prompt_version, json, markdown,"
        " created_at) VALUES ('day', ?, 'test', ?, '{}', ?, ?)",
        (day, PROMPT_VERSION, "# digest\n", now),
    )
    conn.commit()

    cfg.paths.digests.mkdir(parents=True, exist_ok=True)
    (cfg.paths.digests / f"{day}.md").write_text("# digest\n", encoding="utf-8")
    mirror = cfg.paths.digest_mirror
    if mirror is not None:
        mirror.mkdir(parents=True, exist_ok=True)
        (mirror / f"{day}.md").write_text("# digest\n", encoding="utf-8")

    # A not-yet-ingested inbox file whose assumed (rotation-interval) window overlaps
    # the delete range used by the tests below.
    matching = cfg.paths.inbox / "mac_mic_dev1_20260915T010030Z.wav"
    _write_wav(matching, seconds=1)
    (matching.with_suffix(".json")).write_text('{"source": "mac_mic"}', encoding="utf-8")

    # An unrelated inbox file, far outside the range - must survive.
    unrelated = cfg.paths.inbox / "mac_mic_dev1_20260916T010000Z.wav"
    _write_wav(unrelated, seconds=1)

    # A `.part` file (recorder still writing), timestamped inside the same window -
    # never touched, regardless of the timestamp in its name.
    part = cfg.paths.inbox / "mac_mic_dev1_20260915T010045Z.wav.part"
    _write_wav(part, seconds=1)

    return {
        "day": day,
        "rec_id": rec_id,
        "episode_id": episode_id,
        "episode_key": episode_key,
        "started_at_utc": started_at_utc,
        "matching_name": matching.name,
        "unrelated_name": unrelated.name,
        "part_name": part.name,
    }


class TestDeleteRange:
    def test_dry_run_touches_nothing(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg, conn = vas
        cfg.paths.digest_mirror_dir = tmp_path / "mirror"
        cfg.ensure_dirs()
        info = _setup_day_for_delete_range(cfg, conn, monkeypatch)

        start_utc = "2026-09-15T01:00:00Z"
        end_utc = "2026-09-15T01:01:00Z"
        report = delete_range(conn, cfg, start_utc, end_utc, dry_run=True)

        assert report.dry_run is True
        assert len(report.recordings) == 1
        assert report.recordings[0]["id"] == info["rec_id"]

        # Nothing on disk or in the DB actually changed.
        assert (
            conn.execute("SELECT id FROM recordings WHERE id = ?", (info["rec_id"],)).fetchone()
            is not None
        )
        assert conn.execute("SELECT COUNT(*) AS n FROM summaries").fetchone()["n"] == 2
        assert (cfg.paths.digests / f"{info['day']}.md").is_file()
        assert (cfg.paths.inbox / info["matching_name"]).exists()
        assert search(conn, "削除される発言です") != []

    def test_real_run_removes_everything_it_should_and_nothing_else(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg, conn = vas
        cfg.paths.digest_mirror_dir = tmp_path / "mirror"
        cfg.ensure_dirs()
        info = _setup_day_for_delete_range(cfg, conn, monkeypatch)

        audio_path = (
            cfg.paths.store
            / conn.execute(
                "SELECT storage_path FROM recordings WHERE id = ?", (info["rec_id"],)
            ).fetchone()["storage_path"]
        )
        assert audio_path.exists()

        start_utc = "2026-09-15T01:00:00Z"
        end_utc = "2026-09-15T01:01:00Z"
        report = delete_range(conn, cfg, start_utc, end_utc, dry_run=False)

        assert report.dry_run is False
        assert info["day"] in report.days

        # Recording row + audio gone.
        assert (
            conn.execute("SELECT id FROM recordings WHERE id = ?", (info["rec_id"],)).fetchone()
            is None
        )
        assert not audio_path.exists()

        # FTS stays consistent: the deleted utterance is not findable any more.
        assert search(conn, "削除される発言です") == []

        # Episode + day summaries gone.
        assert (
            conn.execute(
                "SELECT 1 FROM summaries WHERE scope='episode' AND scope_key=?",
                (info["episode_key"],),
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM summaries WHERE scope='day' AND scope_key=?", (info["day"],)
            ).fetchone()
            is None
        )

        # Episodes rebuilt for the day: the old episode_id no longer exists (no
        # utterances left to build it from).
        assert (
            conn.execute("SELECT 1 FROM episodes WHERE id = ?", (info["episode_id"],)).fetchone()
            is None
        )

        # Digest file + mirror gone.
        assert not (cfg.paths.digests / f"{info['day']}.md").exists()
        assert not (cfg.paths.digest_mirror / f"{info['day']}.md").exists()

        # The matching inbox file + its sidecar are gone.
        assert not (cfg.paths.inbox / info["matching_name"]).exists()
        assert not (cfg.paths.inbox / info["matching_name"]).with_suffix(".json").exists()

        # The unrelated file and the `.part` file must survive untouched.
        assert (cfg.paths.inbox / info["unrelated_name"]).exists()
        assert (cfg.paths.inbox / info["part_name"]).exists()


# --- timeutil.local_time_to_utc --------------------------------------------------
# (no dedicated tests/test_timeutil.py exists yet, so these live alongside delete_range,
# the feature that needed the function.)


class TestLocalTimeToUtc:
    def test_midnight(self) -> None:
        from voice_ai_summary.timeutil import local_time_to_utc

        assert local_time_to_utc("2026-09-15", "00:00", "Asia/Tokyo") == "2026-09-14T15:00:00Z"

    def test_24_00_means_the_start_of_the_next_local_day(self) -> None:
        from voice_ai_summary.timeutil import local_time_to_utc

        assert local_time_to_utc("2026-09-15", "24:00", "Asia/Tokyo") == "2026-09-15T15:00:00Z"
        # Same instant as 00:00 the next day.
        assert local_time_to_utc("2026-09-15", "24:00", "Asia/Tokyo") == local_time_to_utc(
            "2026-09-16", "00:00", "Asia/Tokyo"
        )

    def test_asia_tokyo_has_no_dst_so_the_offset_is_always_plus_9(self) -> None:
        from voice_ai_summary.timeutil import local_time_to_utc

        assert local_time_to_utc("2026-01-15", "10:30", "Asia/Tokyo") == "2026-01-15T01:30:00Z"
        assert local_time_to_utc("2026-07-15", "10:30", "Asia/Tokyo") == "2026-07-15T01:30:00Z"

    def test_bad_input_raises(self) -> None:
        from voice_ai_summary.timeutil import local_time_to_utc

        with pytest.raises(ValueError):
            local_time_to_utc("2026-09-15", "25:00", "Asia/Tokyo")
        with pytest.raises(ValueError):
            local_time_to_utc("2026-09-15", "10:60", "Asia/Tokyo")
        with pytest.raises(ValueError):
            local_time_to_utc("2026-09-15", "not-a-time", "Asia/Tokyo")
        with pytest.raises(ValueError):
            local_time_to_utc("not-a-date", "10:00", "Asia/Tokyo")
