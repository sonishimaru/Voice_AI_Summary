"""Tests for the VAD→ASR→SQLite pipeline and search."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voice_ai_summary import vad as vad_module
from voice_ai_summary.asr import FakeBackend
from voice_ai_summary.config import Config
from voice_ai_summary.ingest import ingest_file
from voice_ai_summary.pipeline import process_recording, speaker_for_source
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

    backend = FakeBackend(["x"])
    with pytest.raises(FileNotFoundError):
        process_recording(conn, cfg, rec_id, backend)

    row = conn.execute("SELECT error FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert row["error"]


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
        "UPDATE recordings SET error = 'boom', storage_path = ? WHERE id = ?",
        (str(part.relative_to(cfg.paths.store)), rec_id),
    )
    conn.commit()

    assert retry_failed(conn, cfg) == [rec_id]
    assert stored.exists() and not part.exists()
    row = conn.execute(
        "SELECT error, processed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is None and row["processed_at"] is None

    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, None)]
    )
    assert process_pending(conn, cfg, FakeBackend(["再処理"])) == 1
    assert conn.execute("SELECT text FROM utterances").fetchone()["text"] == "再処理"


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

    assert reset_recordings(conn, [rec_id]) == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"] == 0
    assert process_pending(conn, cfg, FakeBackend(["二回目"])) == 1
    assert conn.execute("SELECT text FROM utterances").fetchone()["text"] == "二回目"


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
