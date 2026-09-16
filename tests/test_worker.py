"""Worker shutdown behaviour: a stop signal is not a transcription failure."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voice_ai_summary import vad as vad_module
from voice_ai_summary import worker
from voice_ai_summary.config import Config
from voice_ai_summary.ingest import ingest_file
from voice_ai_summary.pipeline import process_recording
from voice_ai_summary.vad import SpeechRegion


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(seconds * sample_rate))


class _RaisingBackend:
    """Fails the way the worker sees failures: from inside `transcribe`."""

    name = "raising"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def transcribe(self, samples: object, language: str | None = None) -> list:
        raise self._exc


def _one_recording(vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch) -> tuple:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )
    return cfg, conn, rec_id


def test_stop_does_not_become_a_recording_error(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`launchctl kickstart -k` SIGTERMs the worker on every restart, so a stop landing
    mid-transcription is the common path. It must not blame the recording."""
    cfg, conn, rec_id = _one_recording(vas, monkeypatch)

    with pytest.raises(worker._Stop):
        process_recording(conn, cfg, rec_id, _RaisingBackend(worker._Stop()))

    row = conn.execute(
        "SELECT error, processed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is None
    assert row["processed_at"] is None


def test_a_real_failure_is_still_recorded(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, conn, rec_id = _one_recording(vas, monkeypatch)

    with pytest.raises(ValueError):
        process_recording(conn, cfg, rec_id, _RaisingBackend(ValueError("decode failed")))

    row = conn.execute("SELECT error FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert "decode failed" in row["error"]
