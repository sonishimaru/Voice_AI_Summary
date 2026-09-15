"""CLI smoke tests using typer's CliRunner and the fake ASR backend."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voice_ai_summary.cli import app

runner = CliRunner()


def _write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VAS_ASR_BACKEND", "fake")
    return tmp_path


def test_status_runs(env: Path) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "data_dir" in result.output


def test_ingest_process_search_roundtrip(env: Path) -> None:
    src = env / "sample_dev1_20260915T010203Z.wav"
    _write_wav(src, seconds=2.0)

    result = runner.invoke(app, ["ingest", str(src), "--copy"])
    assert result.exit_code == 0, result.output
    assert "recording" in result.output

    result = runner.invoke(app, ["process"])
    assert result.exit_code == 0, result.output
    assert "processed" in result.output

    result = runner.invoke(app, ["search", "テスト"])
    assert result.exit_code == 0, result.output


def test_worker_once(env: Path) -> None:
    src = env / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(src, seconds=1.0)

    result = runner.invoke(app, ["ingest", str(src), "--copy"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["worker", "--once"])
    assert result.exit_code == 0, result.output
