"""Tests for reading the macOS recorder app's state file and event log."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from voice_ai_summary.recorder_state import (
    EVENTS_FILENAME,
    STATE_FILENAME,
    describe,
    format_intervals,
    is_alive,
    pause_intervals,
    read_events,
)

NOW = datetime(2026, 9, 17, 10, 2, 0, tzinfo=UTC)


def _write_state(root: Path, **overrides: object) -> None:
    data = {
        "schema": 1,
        "state": "recording",
        "since": "2026-09-17T09:12:00Z",
        "resume_at": None,
        "reason": "user",
        "pid": os.getpid(),
        "app_version": "0.2.0",
        "updated_at": "2026-09-17T10:02:00Z",
    }
    data.update(overrides)
    (root / STATE_FILENAME).write_text(json.dumps(data), encoding="utf-8")


def _write_events(root: Path, lines: list[str]) -> None:
    (root / EVENTS_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def _event_json(**overrides: object) -> str:
    data = {
        "state": "paused",
        "since": "2026-09-17T10:00:00Z",
        "resume_at": None,
        "reason": "user",
        "pid": os.getpid(),
        "app_version": "0.2.0",
        "updated_at": "2026-09-17T10:00:00Z",
    }
    data.update(overrides)
    return json.dumps(data)


# --- read_state ---------------------------------------------------------------


def test_read_state_missing_file_returns_none(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    assert read_state(tmp_path) is None


def test_read_state_invalid_json_returns_none(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    (tmp_path / STATE_FILENAME).write_text("{not json", encoding="utf-8")
    assert read_state(tmp_path) is None


def test_read_state_wrong_schema_returns_none(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(tmp_path, schema=2)
    assert read_state(tmp_path) is None


def test_read_state_missing_required_key_returns_none(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    data = {"schema": 1, "state": "recording"}
    (tmp_path / STATE_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    assert read_state(tmp_path) is None


def test_read_state_valid_file_parses(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(tmp_path)
    state = read_state(tmp_path)
    assert state is not None
    assert state.state == "recording"
    assert state.pid == os.getpid()
    assert state.app_version == "0.2.0"


# --- is_alive ------------------------------------------------------------------


def test_is_alive_true_for_own_pid_and_fresh_heartbeat(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(tmp_path, pid=os.getpid(), updated_at="2026-09-17T10:02:00Z")
    state = read_state(tmp_path)
    assert state is not None
    assert is_alive(state, now=NOW) is True


def test_is_alive_false_for_nonexistent_pid(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(tmp_path, pid=999_999_999, updated_at="2026-09-17T10:02:00Z")
    state = read_state(tmp_path)
    assert state is not None
    assert is_alive(state, now=NOW) is False


def test_is_alive_false_when_kill_raises_process_lookup_error(
    tmp_path: Path, monkeypatch: object
) -> None:
    from voice_ai_summary import recorder_state
    from voice_ai_summary.recorder_state import read_state

    def fake_kill(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(recorder_state.os, "kill", fake_kill)
    _write_state(tmp_path, pid=4321, updated_at="2026-09-17T10:02:00Z")
    state = read_state(tmp_path)
    assert state is not None
    assert is_alive(state, now=NOW) is False


def test_is_alive_false_for_stale_heartbeat_even_with_live_pid(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    # 400s before `now`, well past HEARTBEAT_STALE_S (300s).
    _write_state(tmp_path, pid=os.getpid(), updated_at="2026-09-17T09:55:20Z")
    state = read_state(tmp_path)
    assert state is not None
    assert is_alive(state, now=NOW) is False


# --- describe --------------------------------------------------------------------


def test_describe_none_state() -> None:
    assert describe(None, "UTC", now=NOW) == (
        "unknown (no recorder state file; the recorder app predates this feature or has never run)"
    )


def test_describe_recording_alive(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="recording",
        since="2026-09-17T09:12:00Z",
        pid=os.getpid(),
        updated_at="2026-09-17T10:02:00Z",
    )
    state = read_state(tmp_path)
    assert state is not None
    assert describe(state, "UTC", now=NOW) == f"recording since 09:12 (pid {os.getpid()})"


def test_describe_recording_stale(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="recording",
        since="2026-09-17T09:12:00Z",
        pid=os.getpid(),
        updated_at="2026-09-17T09:40:00Z",  # stale: 22 min before `now`
    )
    state = read_state(tmp_path)
    assert state is not None
    assert describe(state, "UTC", now=NOW) == (
        "state file says recording since 09:12 but the process is gone or stale "
        "(last heartbeat 09:40)"
    )


def test_describe_paused_with_resume_at_minutes_left(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="paused",
        since="2026-09-17T10:00:00Z",
        resume_at="2026-09-17T10:30:00Z",
        updated_at="2026-09-17T10:00:00Z",
    )
    state = read_state(tmp_path)
    assert state is not None
    # now = 10:02, resume_at = 10:30 -> 28 minutes left.
    assert describe(state, "UTC", now=NOW) == "paused since 10:00, resumes 10:30 (28 min left)"


def test_describe_paused_resume_time_passed(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="paused",
        since="2026-09-17T10:00:00Z",
        resume_at="2026-09-17T10:30:00Z",
        updated_at="2026-09-17T10:00:00Z",
    )
    state = read_state(tmp_path)
    assert state is not None
    later = datetime(2026, 9, 17, 10, 35, 0, tzinfo=UTC)
    assert describe(state, "UTC", now=later) == (
        "resume time passed at 10:30 — the app should have resumed"
    )


def test_describe_paused_without_resume_at(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="paused",
        since="2026-09-17T10:00:00Z",
        resume_at=None,
        updated_at="2026-09-17T10:00:00Z",
    )
    state = read_state(tmp_path)
    assert state is not None
    assert describe(state, "UTC", now=NOW) == "paused since 10:00 until resumed by hand"


def test_describe_stopped(tmp_path: Path) -> None:
    from voice_ai_summary.recorder_state import read_state

    _write_state(
        tmp_path,
        state="stopped",
        since="2026-09-17T18:00:00Z",
        resume_at=None,
        updated_at="2026-09-17T18:00:00Z",
    )
    state = read_state(tmp_path)
    assert state is not None
    assert describe(state, "UTC", now=NOW) == "stopped since 18:00"


# --- read_events -----------------------------------------------------------------


def test_read_events_skips_torn_line_and_delete_recent(tmp_path: Path) -> None:
    lines = [
        _event_json(state="recording", reason="launch", updated_at="2026-09-17T08:00:00Z"),
        _event_json(state="paused", reason="user", updated_at="2026-09-17T10:00:00Z"),
        _event_json(
            state="paused",
            reason="delete_recent",
            updated_at="2026-09-17T10:05:00Z",
            minutes=10,
            files=2,
        ),
        _event_json(state="recording", reason="user", updated_at="2026-09-17T10:30:00Z"),
        '{"state": "paused", "reason": "timer", "upda',  # torn last line
    ]
    _write_events(tmp_path, lines)

    events = read_events(tmp_path, "2026-09-17T09:00:00Z", "2026-09-17T11:00:00Z")
    assert [(e.state, e.at, e.reason) for e in events] == [
        ("paused", "2026-09-17T10:00:00Z", "user"),
        ("recording", "2026-09-17T10:30:00Z", "user"),
    ]


# --- pause_intervals -------------------------------------------------------------


def test_pause_intervals_paused_then_recording_inside_window(tmp_path: Path) -> None:
    lines = [
        _event_json(state="paused", reason="user", updated_at="2026-09-17T10:00:00Z"),
        _event_json(state="recording", reason="user", updated_at="2026-09-17T10:30:00Z"),
    ]
    _write_events(tmp_path, lines)

    intervals = pause_intervals(tmp_path, "2026-09-17T09:00:00Z", "2026-09-17T12:00:00Z", now=NOW)
    assert intervals == [("2026-09-17T10:00:00Z", "2026-09-17T10:30:00Z")]


def test_pause_intervals_open_interval_closes_at_now(tmp_path: Path) -> None:
    lines = [
        _event_json(state="paused", reason="user", updated_at="2026-09-17T10:00:00Z"),
    ]
    _write_events(tmp_path, lines)

    now = datetime(2026, 9, 17, 10, 45, 0, tzinfo=UTC)
    intervals = pause_intervals(tmp_path, "2026-09-17T09:00:00Z", "2026-09-17T12:00:00Z", now=now)
    assert intervals == [("2026-09-17T10:00:00Z", "2026-09-17T10:45:00Z")]


def test_pause_intervals_stopped_then_recording(tmp_path: Path) -> None:
    lines = [
        _event_json(state="stopped", reason="quit", updated_at="2026-09-17T14:00:00Z"),
        _event_json(state="recording", reason="launch", updated_at="2026-09-17T14:20:00Z"),
    ]
    _write_events(tmp_path, lines)

    intervals = pause_intervals(tmp_path, "2026-09-17T13:00:00Z", "2026-09-17T15:00:00Z", now=NOW)
    assert intervals == [("2026-09-17T14:00:00Z", "2026-09-17T14:20:00Z")]


def test_pause_intervals_clips_interval_straddling_window_start(tmp_path: Path) -> None:
    lines = [
        _event_json(state="paused", reason="user", updated_at="2026-09-17T09:50:00Z"),
        _event_json(state="recording", reason="user", updated_at="2026-09-17T10:05:00Z"),
    ]
    _write_events(tmp_path, lines)

    intervals = pause_intervals(tmp_path, "2026-09-17T10:00:00Z", "2026-09-17T12:00:00Z", now=NOW)
    assert intervals == [("2026-09-17T10:00:00Z", "2026-09-17T10:05:00Z")]


def test_pause_intervals_no_events_but_state_file_paused_before_window(
    tmp_path: Path,
) -> None:
    _write_state(
        tmp_path,
        state="paused",
        since="2026-09-17T08:00:00Z",
        resume_at=None,
        updated_at="2026-09-17T08:00:00Z",
    )
    # No recorder_events.jsonl at all.
    later = datetime(2026, 9, 17, 13, 0, 0, tzinfo=UTC)
    intervals = pause_intervals(tmp_path, "2026-09-17T10:00:00Z", "2026-09-17T12:00:00Z", now=later)
    assert intervals == [("2026-09-17T10:00:00Z", "2026-09-17T12:00:00Z")]


def test_pause_intervals_merges_adjacent_intervals(tmp_path: Path) -> None:
    lines = [
        _event_json(state="paused", reason="user", updated_at="2026-09-17T10:00:00Z"),
        _event_json(state="recording", reason="user", updated_at="2026-09-17T10:15:00Z"),
        _event_json(state="paused", reason="user", updated_at="2026-09-17T10:15:00Z"),
        _event_json(state="recording", reason="user", updated_at="2026-09-17T10:30:00Z"),
    ]
    _write_events(tmp_path, lines)

    intervals = pause_intervals(tmp_path, "2026-09-17T09:00:00Z", "2026-09-17T12:00:00Z", now=NOW)
    assert intervals == [("2026-09-17T10:00:00Z", "2026-09-17T10:30:00Z")]


# --- format_intervals --------------------------------------------------------------


def test_format_intervals_in_asia_tokyo() -> None:
    intervals = [
        ("2026-09-17T01:00:00Z", "2026-09-17T01:30:00Z"),
        ("2026-09-17T05:05:00Z", "2026-09-17T05:20:00Z"),
    ]
    assert format_intervals(intervals, "Asia/Tokyo") == "10:00–10:30, 14:05–14:20"


def test_format_intervals_empty() -> None:
    assert format_intervals([], "UTC") == ""
