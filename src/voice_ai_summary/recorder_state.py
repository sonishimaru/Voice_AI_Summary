"""Read the macOS recorder app's state file and event log.

The recorder (a separate Swift app) writes two files under the data directory:

- `recorder_state.json`: the current state, replaced atomically on every transition
  and, while recording, rewritten every 60s as a heartbeat.
- `recorder_events.jsonl`: one JSON object appended per transition (plus
  `delete_recent` notices, which are not transitions).

This module is read-only and defensive: any malformed input here means the recorder
predates this feature, has never run, or wrote a torn file mid-update, never a reason
to raise out of a status/report code path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from .timeutil import fmt_hm

STATE_FILENAME = "recorder_state.json"
EVENTS_FILENAME = "recorder_events.jsonl"

# The recorder rewrites its heartbeat every 60s while recording; five missed
# heartbeats in a row means the process is gone, not just briefly slow.
HEARTBEAT_STALE_S = 300

_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_TIME_FMT)


def _parse_utc(iso_utc: str) -> datetime | None:
    try:
        return datetime.strptime(iso_utc, _TIME_FMT).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class RecorderState:
    state: str  # recording | paused | stopped
    since: str
    resume_at: str | None
    reason: str  # user | timer | launch | sleep | wake | retry | quit | delete_recent
    pid: int | None
    app_version: str
    updated_at: str


@dataclass(frozen=True)
class Event:
    state: str
    at: str  # the event's `updated_at`, used as its timestamp
    reason: str
    resume_at: str | None


def read_state(root: Path) -> RecorderState | None:
    """Read `recorder_state.json` under `root`.

    Returns None (never raises) for a missing file, an unreadable one, invalid JSON,
    an unrecognised `schema`, or a JSON object missing a required key - all of which
    mean the recorder predates this feature or has never run.
    """
    try:
        text = (root / STATE_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("schema") != 1:
        return None
    # `resume_at` is deliberately not required: the app now writes `"resume_at": null`
    # explicitly, but a synthesized Swift encoder omits a nil Optional entirely, and a
    # file from a build that did that must still parse.
    required = ("state", "since", "reason", "pid", "app_version", "updated_at")
    if not all(key in data for key in required):
        return None
    return RecorderState(
        state=data["state"],
        since=data["since"],
        resume_at=data.get("resume_at"),
        reason=data["reason"],
        pid=data["pid"],
        app_version=data["app_version"],
        updated_at=data["updated_at"],
    )


def is_alive(state: RecorderState, *, now: datetime) -> bool:
    """True iff `state.pid` looks like a live process AND its heartbeat is fresh."""
    if state.pid is None:
        return False
    try:
        os.kill(state.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, just not ours - still counts as alive
    updated_dt = _parse_utc(state.updated_at)
    if updated_dt is None:
        return False
    return abs((now - updated_dt).total_seconds()) <= HEARTBEAT_STALE_S


def describe(state: RecorderState | None, tz: str, *, now: datetime) -> str:
    """One human-readable line summarising the recorder's state."""
    if state is None:
        return (
            "unknown (no recorder state file; the recorder app predates this feature "
            "or has never run)"
        )
    since_str = fmt_hm(state.since, tz)
    if state.state == "recording":
        if is_alive(state, now=now):
            return f"recording since {since_str} (pid {state.pid})"
        return (
            f"state file says recording since {since_str} but the process is gone "
            f"or stale (last heartbeat {fmt_hm(state.updated_at, tz)})"
        )
    if state.state == "paused":
        if state.resume_at:
            resume_str = fmt_hm(state.resume_at, tz)
            resume_dt = _parse_utc(state.resume_at)
            if resume_dt is not None and resume_dt > now:
                minutes_left = round((resume_dt - now).total_seconds() / 60)
                return f"paused since {since_str}, resumes {resume_str} ({minutes_left} min left)"
            return f"resume time passed at {resume_str} — the app should have resumed"
        return f"paused since {since_str} until resumed by hand"
    return f"stopped since {since_str}"


def read_all_events(root: Path) -> list[Event]:
    """Parse the entire `recorder_events.jsonl`, tolerating a torn last line.

    `delete_recent` and `heartbeat` entries (not transitions) are skipped, along with
    any line that isn't valid JSON, isn't an object, or is missing a required key.
    The app no longer writes heartbeat lines here; skipping them covers files written
    by an earlier build.
    """
    try:
        text = (root / EVENTS_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return []
    events: list[Event] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("reason") in ("delete_recent", "heartbeat"):
            continue
        if not all(key in data for key in ("state", "reason", "updated_at")):
            continue
        events.append(
            Event(
                state=data["state"],
                at=data["updated_at"],
                reason=data["reason"],
                resume_at=data.get("resume_at"),
            )
        )
    return events


def pause_intervals(
    root: Path, start_utc: str, end_utc: str, *, now: datetime
) -> list[tuple[str, str]]:
    """Intervals within `[start_utc, end_utc)` during which the recorder was not
    recording (paused or stopped), as merged, clipped `(start, end)` UTC ISO pairs.

    Reads the whole event history (not just the window) so an event before the
    window can still establish the state the window opens in.
    """
    events = sorted(read_all_events(root), key=lambda e: e.at)
    now_iso = _fmt_utc(now)

    # (state, segment_start, segment_end_or_None-if-still-open)
    segments: list[tuple[str, str, str | None]] = []
    if events:
        for i, event in enumerate(events):
            seg_end = events[i + 1].at if i + 1 < len(events) else None
            segments.append((event.state, event.at, seg_end))
    else:
        # No transition history at all: fall back to the live state file. If it says
        # paused/stopped since before the window, the whole window is one interval.
        state = read_state(root)
        if state is not None and state.state != "recording" and state.since < start_utc:
            segments.append((state.state, state.since, None))

    intervals: list[tuple[str, str]] = []
    for state_name, seg_start, seg_end in segments:
        if state_name == "recording":
            continue
        end = seg_end if seg_end is not None else min(now_iso, end_utc)
        start = max(seg_start, start_utc)
        end = min(end, end_utc)
        if start < end:
            intervals.append((start, end))

    intervals.sort()
    merged: list[tuple[str, str]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_intervals(intervals: list[tuple[str, str]], tz: str) -> str:
    """Render `pause_intervals` output as e.g. "10:00–10:30, 14:05–14:20"."""
    return ", ".join(f"{fmt_hm(start, tz)}–{fmt_hm(end, tz)}" for start, end in intervals)


_T = TypeVar("_T")


def with_pause_markers(
    items: list[_T],
    intervals: list[tuple[str, str]],
    tz: str,
    *,
    start_of: Callable[[_T], str],
    render: Callable[[_T], str],
) -> list[str]:
    """Interleave `--- recorder paused HH:MM–HH:MM ---` marker lines between rendered
    `items` at the chronologically correct spot.

    `start_of(item)` gives the item's UTC ISO start timestamp, used only to decide
    where each interval falls; `render(item)` gives the line actually emitted for it.
    An interval is emitted right before the first item whose `start_of` is at or past
    that interval's end - i.e. it is placed as early as it can be while still coming
    after every item it overlaps or precedes. Any interval(s) after the last item go
    at the very end. `items` must already be sorted oldest-first.
    """
    lines: list[str] = []
    i = 0
    for item in items:
        item_start = start_of(item)
        while i < len(intervals) and item_start >= intervals[i][1]:
            lines.append(f"--- recorder paused {format_intervals([intervals[i]], tz)} ---")
            i += 1
        lines.append(render(item))
    while i < len(intervals):
        lines.append(f"--- recorder paused {format_intervals([intervals[i]], tz)} ---")
        i += 1
    return lines
