"""CLI smoke tests using typer's CliRunner and the fake ASR backend."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import patch

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


_FAKE_DIGEST_MARKDOWN = (
    "# 2026-09-15 の記録\n\n"
    "## ハイライト\n"
    "- 打ち合わせでリリース日を決定\n\n"
    "## 本文\n"
    "この行はハイライトより下の本文で、ヘッドラインには出ない一意な文字列です。\n"
)


def test_digest_without_flags_prints_path_and_headline_not_body(env: Path) -> None:
    """Under launchd, `vas digest` (no --deliver) runs nightly whenever no channel is
    enabled; the full markdown must not land in the log file, only a path + headline."""
    with patch("voice_ai_summary.summarize.run_day", return_value=_FAKE_DIGEST_MARKDOWN):
        result = runner.invoke(app, ["digest", "--day", "2026-09-15"])
    assert result.exit_code == 0, result.output
    assert f"wrote {env / 'data' / 'digests' / '2026-09-15.md'}" in result.output
    assert "打ち合わせでリリース日を決定" in result.output  # the headline bullet
    # The full body must not land verbatim (this is what a launchd log would show).
    assert "ヘッドラインには出ない一意な文字列" not in result.output


def test_digest_show_prints_the_full_markdown(env: Path) -> None:
    with patch("voice_ai_summary.summarize.run_day", return_value=_FAKE_DIGEST_MARKDOWN):
        result = runner.invoke(app, ["digest", "--day", "2026-09-15", "--show"])
    assert result.exit_code == 0, result.output
    assert "ヘッドラインには出ない一意な文字列" in result.output
    assert "wrote " not in result.output


def test_harden_dry_run_lists_a_change_and_changes_nothing(env: Path) -> None:
    import stat

    data_dir = env / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.chmod(0o755)

    result = runner.invoke(app, ["harden", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "0755 -> 0700" in result.output
    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o755  # unchanged


def test_install_launchd_output_mentions_hardened(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("voice_ai_summary.launchd.sys.platform", "linux")
    monkeypatch.setattr("pathlib.Path.home", lambda: env)

    result = runner.invoke(app, ["install-launchd"])
    assert result.exit_code == 0, result.output
    assert "hardened" in result.output
