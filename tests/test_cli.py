"""CLI smoke tests using typer's CliRunner and the fake ASR backend."""

from __future__ import annotations

import json
import os
import wave
from datetime import UTC, datetime, timedelta
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


def test_status_recorder_unknown_without_state_file(env: Path) -> None:
    """No `recorder_state.json` at all: the recorder app predates this feature or has
    never run, so `status` says so rather than showing anything misleading."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "recorder : unknown" in result.output
    assert "audio-pruned recordings: 0" in result.output


def test_status_shows_recorder_state_and_audio_pruned_count(env: Path) -> None:
    data_dir = env / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    state = {
        "schema": 1,
        "state": "recording",
        "since": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resume_at": None,
        "reason": "user",
        "pid": os.getpid(),
        "app_version": "0.2.0",
        "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (data_dir / "recorder_state.json").write_text(json.dumps(state), encoding="utf-8")

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "recorder : recording since" in result.output
    assert "audio-pruned recordings: 0" in result.output


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


# --- `vas episodes` + recorder-paused markers -------------------------------------


def _insert_recording(conn, *, source: str, started_at_utc: str, sha256: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at
        ) VALUES (?, 'dev1', ?, '+00:00', 1000, ?, 'store/x.wav', 'x.wav', ?, ?)
        """,
        (source, started_at_utc, sha256, started_at_utc, started_at_utc),
    )
    return cur.lastrowid


def _insert_utterance(
    conn,
    *,
    recording_id: int,
    rec_started_at_utc: str,
    t_start_ms: int,
    t_end_ms: int,
    speaker: str,
    text: str = "テスト発話",
) -> int:
    base = datetime.strptime(rec_started_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    abs_start = (base + timedelta(milliseconds=t_start_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        INSERT INTO utterances(
            recording_id, t_start_ms, t_end_ms, abs_start_utc, text, lang,
            asr_model, avg_logprob, speaker
        ) VALUES (?, ?, ?, ?, ?, 'ja', 'fake', -0.1, ?)
        """,
        (recording_id, t_start_ms, t_end_ms, abs_start, text, speaker),
    )
    return cur.lastrowid


def _make_two_episodes(conn) -> None:
    """Two well-separated groups of utterances on 2026-09-15 (JST), split by
    `build_episodes` into two episodes with a 10-minute gap between them."""
    started = "2026-09-15T00:00:00Z"
    rec_id = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="a" * 64)
    conn.commit()

    for offset in (0, 10_000, 20_000, 30_000):
        _insert_utterance(
            conn,
            recording_id=rec_id,
            rec_started_at_utc=started,
            t_start_ms=offset,
            t_end_ms=offset + 5_000,
            speaker="me",
        )
    # Gap of 10 minutes (> default 5 minute episodes.gap_minutes) before group 2.
    group2_start = 35_000 + 10 * 60_000
    for i in range(4):
        offset = group2_start + i * 10_000
        _insert_utterance(
            conn,
            recording_id=rec_id,
            rec_started_at_utc=started,
            t_start_ms=offset,
            t_end_ms=offset + 5_000,
            speaker="me",
        )
    conn.commit()


def test_episodes_no_state_file_output_unchanged(env: Path) -> None:
    """No recorder state/events file at all: `episodes` prints exactly what it did
    before this feature - no paused markers appear."""
    result = runner.invoke(app, ["episodes", "--day", "2026-09-15"])
    assert result.exit_code == 0, result.output
    assert result.output == "2026-09-15: no episodes\n"


def test_episodes_shows_paused_marker_between_episodes(env: Path) -> None:
    from voice_ai_summary.config import load_config
    from voice_ai_summary.db import connect

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    _make_two_episodes(conn)
    conn.close()

    # A pause that starts after episode 1 ends (00:00:35) and ends before episode 2
    # starts (00:10:35), well inside the 10-minute gap between them.
    events = [
        json.dumps({"state": "paused", "reason": "user", "updated_at": "2026-09-15T00:01:00Z"}),
        json.dumps({"state": "recording", "reason": "user", "updated_at": "2026-09-15T00:05:00Z"}),
    ]
    (cfg.paths.root / "recorder_events.jsonl").write_text("\n".join(events), encoding="utf-8")

    result = runner.invoke(app, ["episodes", "--day", "2026-09-15"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()

    episode_line_indices = [i for i, line in enumerate(lines) if line.startswith("#")]
    paused_line_indices = [
        i for i, line in enumerate(lines) if line.startswith("--- recorder paused")
    ]
    assert len(episode_line_indices) == 2
    assert len(paused_line_indices) == 1
    # The marker sits chronologically between the two episodes, not before or after both.
    assert episode_line_indices[0] < paused_line_indices[0] < episode_line_indices[1]
    assert "09:01" in lines[paused_line_indices[0]]  # JST local time for 00:01Z
    assert "09:05" in lines[paused_line_indices[0]]


def test_episodes_no_episodes_still_shows_paused_marker(env: Path) -> None:
    from voice_ai_summary.config import load_config

    cfg = load_config()
    cfg.ensure_dirs()

    events = [
        json.dumps({"state": "paused", "reason": "user", "updated_at": "2026-09-15T00:01:00Z"}),
        json.dumps({"state": "recording", "reason": "user", "updated_at": "2026-09-15T00:05:00Z"}),
    ]
    (cfg.paths.root / "recorder_events.jsonl").write_text("\n".join(events), encoding="utf-8")

    result = runner.invoke(app, ["episodes", "--day", "2026-09-15"])
    assert result.exit_code == 0, result.output
    assert "2026-09-15: no episodes" in result.output
    assert "--- recorder paused" in result.output


def test_prune_cmd_truncates_worker_log_without_skip_logs(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`vas prune` runs as its own short-lived process, separate from the worker - it
    must not pass `skip_logs` (that's `worker.run_worker`'s job alone), so it truncates
    even a log named after the worker's own launchd log."""
    from voice_ai_summary import launchd

    monkeypatch.setattr(Path, "home", lambda: env)
    config_path = env / "config.toml"
    config_path.write_text("[retention]\nlog_max_bytes = 50\n", encoding="utf-8")
    monkeypatch.setenv("VAS_CONFIG", str(config_path))

    log_dir = launchd.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{launchd.WORKER_LABEL}.log"
    log_path.write_text("x" * 500, encoding="utf-8")

    result = runner.invoke(app, ["prune", "--yes"])

    assert result.exit_code == 0, result.output
    assert log_path.name in result.output
    assert log_path.stat().st_size < 500
