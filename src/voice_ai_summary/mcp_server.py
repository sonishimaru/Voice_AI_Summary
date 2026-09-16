"""MCP stdio server exposing `vas` functionality as tools for Claude Desktop.

This lets the person driving `vas` do so entirely from Claude Desktop instead of a
terminal: read digests, search/browse transcripts, rebuild a day's summary, manage the
personal glossary, check pipeline status and API spend, retry failed recordings, and
pull+reinstall the latest code.

IMPORTANT: stdout carries the MCP JSON-RPC protocol itself. Nothing in this module (or
anything it imports) may `print` or otherwise write to stdout - that would corrupt the
stream and break the session. Diagnostics belong on stderr via `logging`, which is safe.

Heavy imports (config, db, summarize, ...) are kept lazy inside each tool function, the
same style used by `commands_*.py`, so importing this module (which `mcp_server.main`
does at process start) stays cheap.
"""

from __future__ import annotations

import functools
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from mcp.server.mcpserver import MCPServer

if TYPE_CHECKING:
    from .asr import ASRBackend
    from .config import Config

server = MCPServer(
    name="voice-ai-summary",
    instructions=(
        "Tools for the `vas` voice-ai-summary pipeline: an always-on voice recorder that "
        "locally transcribes Japanese speech and uses Claude to produce daily digests. "
        "Use `daily_summary`/`list_days` to read what has already been summarized, "
        "`search_transcript`/`transcript` to look at the raw local transcript, "
        '`recent` to see the last few minutes of transcript ("what was just said") - '
        "note this can lag the live conversation by several minutes because of file "
        "rotation, and the tool's header says by how much, "
        "`rebuild_day` to (re)run summarization for a day - this calls the Claude API "
        "and costs money, and for a large day can take several minutes, well past "
        "this session's tool-call timeout, so it starts the work in the background "
        "and returns a job id immediately instead of waiting; follow up with "
        "`job_status` (by job id, or with no id to list recent jobs) to see progress "
        "and get the resulting digest or error once it finishes - calling "
        "`rebuild_day` again for a day already running just hands back the same job "
        "instead of paying for the work twice, "
        "`status`/`api_usage` to check pipeline health and spend, "
        "`throughput` to check whether local transcription is running faster or slower "
        "than realtime (e.g. after switching ASR backend) and see the pending backlog's "
        "estimated time to clear, "
        "`list_vocabulary`/`add_vocabulary` to manage the personal glossary, "
        "`retry_failed` to re-queue recordings that failed to transcribe and "
        "`process_pending` to actually transcribe pending recordings by hand (local, "
        "free, one small batch per call), `list_recordings` to see the raw recording "
        "rows for a day (source, time, duration, utterance count, state, sha256, "
        "storage path) - this is how two recordings covering the same time span show "
        "up, `find_duplicates` to find utterances whose text repeats within a day and "
        "which recording ids that duplication comes from, `drop_recording` to "
        "permanently delete one recording's database rows (never the audio file) once "
        "a duplicate has been identified, `worker_status` to see why the background "
        "worker or nightly digest isn't running, `restart_worker`/`install_services` to "
        "fix it, `service_logs` to see why it crashed, and `update_app` to pull and "
        "reinstall the latest `vas` code."
    ),
)

F = TypeVar("F", bound=Callable[..., str])


def _tool_safe(func: F) -> F:
    """Wrap a tool body so any exception it raises becomes a returned string instead of
    propagating and killing the stdio session.

    `RuntimeError` (which `llm.BudgetExceeded` subclasses, and which tools raise
    themselves for user-facing problems) is surfaced verbatim. Anything else becomes
    `f"{type(exc).__name__}: {exc}"`, so a real bug is still legible instead of silently
    swallowed.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> str:
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - this IS the tool-boundary catch-all
            return f"{type(exc).__name__}: {exc}"

    return wrapper  # type: ignore[return-value]


# --- Background job registry -----------------------------------------------------
#
# Claude Desktop cuts a tool call off at roughly 60 seconds, but `rebuild_day` can run
# for minutes on a large day. Rather than let the call time out while the (paid) work
# keeps going invisibly - and risk the user retrying it, doubling the API spend - long
# tools run on a background daemon thread and hand back a `Job` id immediately;
# `job_status` polls it. Kept deliberately simple: one dict of `Job`s guarded by one
# lock, no persistence (jobs vanish on server restart, same as anything else in this
# process).


@dataclass
class Job:
    """One background job's state. Mutated only while holding `_jobs_lock`."""

    id: str
    kind: str
    key: str
    started_at: float
    state: str = "running"  # "running" | "done" | "failed"
    progress: str = "running"
    finished_at: float | None = None
    result: str | None = None
    error: str | None = None
    # Whether this job was started with `force=True` (currently only meaningful for
    # `rebuild_day`). Recorded on the job itself, not just passed to `fn`, so
    # `job_status` can say plainly whether the job actually running under a given id is
    # doing the forced recompute or not - a caller who asked for `force=True` and got
    # back a *different* job's handle (see `_start_job`) must be able to tell that job
    # is not forced, rather than assuming its own force request took effect.
    force: bool = False


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# Cap how many recent jobs `job_status()` (no id) lists, so a long-lived server doesn't
# dump an ever-growing history.
_JOB_LIST_LIMIT = 20


def _start_job(
    kind: str, key: str, fn: Callable[[Job], str], *, force: bool = False
) -> tuple[Job, bool]:
    """Start `fn(job)` on a background daemon thread and return `(job, started)`
    immediately, without waiting for it to finish.

    Guards one job per `(kind, key)` at a time: if a job with the same kind and key is
    already running, that job is returned with `started=False` and no new thread is
    started - this is what stops a Claude-Desktop-timeout-triggered retry of the same
    tool call from doubling a paid Claude API workload, and also what stops a `force=True`
    request from starting a second, concurrent, paid rebuild of a day that a non-forced
    job is already rebuilding. The check-and-create is one atomic section under
    `_jobs_lock`, so two concurrent callers can never both "win".

    Note that the *existing* job's own `force` may not match the `force` this call was
    asked for - `_start_job` never launches a second job to reconcile that, it only
    reports what is actually running (via the returned `Job.force`) so the caller can
    tell the difference and say so honestly instead of implying the requested work is
    underway when it is not.

    `fn` receives the `Job` itself (so it can call `_set_progress` as it goes) and
    should return the job's result string. Any exception `fn` raises is caught and
    captured onto the job record (state becomes "failed") rather than propagating -
    daemon threads are otherwise silent on failure, and this would otherwise both hide
    the error from the user and prevent `job_status` from ever reporting it.
    """
    with _jobs_lock:
        for existing in _jobs.values():
            if existing.kind == kind and existing.key == key and existing.state == "running":
                return existing, False
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, key=key, started_at=time.time(), force=force)
        _jobs[job.id] = job

    def _worker() -> None:
        try:
            result = fn(job)
        except Exception as exc:  # noqa: BLE001 - captured onto the job, never re-raised
            with _jobs_lock:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()
            return
        with _jobs_lock:
            job.state = "done"
            job.result = result
            job.finished_at = time.time()

    threading.Thread(target=_worker, daemon=True).start()
    return job, True


def _set_progress(job: Job, text: str) -> None:
    """Update `job`'s latest progress line - called from the job's worker thread."""
    with _jobs_lock:
        job.progress = text


def _elapsed(job: Job) -> str:
    """Human-readable time since `job` started (its total run time, once finished)."""
    end = job.finished_at if job.finished_at is not None else time.time()
    secs = max(0.0, end - job.started_at)
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs / 60:.1f}min"


def _format_job_summary(job: Job) -> str:
    force_note = "  force=True" if job.force else ""
    return f"{job.id}  {job.kind}({job.key}){force_note}  {job.state}  elapsed={_elapsed(job)}"


def _format_job_detail(job: Job) -> str:
    lines = [
        f"job {job.id}: {job.kind}({job.key})",
        f"force: {job.force}",
        f"state: {job.state}",
        f"elapsed: {_elapsed(job)}",
        f"progress: {job.progress}",
    ]
    if job.state == "done":
        lines.append("")
        lines.append("result:")
        lines.append(job.result or "")
    elif job.state == "failed":
        lines.append(f"error: {job.error}")
    return "\n".join(lines)


def _format_utterance_row(row: sqlite3.Row, tz: str) -> str:
    """Render one utterance row as `HH:MM [speaker] text` in local time."""
    from .timeutil import fmt_hm

    return f"{fmt_hm(row['abs_start_utc'], tz)} [{row['speaker']}] {row['text']}"


def _stored_digest(conn: sqlite3.Connection, day: str) -> str | None:
    """The day digest held in `summaries`, or None when it has not been built."""
    from .summarize import PROMPT_VERSION

    row = conn.execute(
        "SELECT markdown FROM summaries WHERE scope='day' AND scope_key=? AND prompt_version=?",
        (day, PROMPT_VERSION),
    ).fetchone()
    return row["markdown"] if row is not None else None


@_tool_safe
def daily_summary(day: str = "") -> str:
    """Return the stored Japanese Markdown daily digest for a local day.

    `day` is a local date `YYYY-MM-DD` in the configured timezone; empty means today.
    Read-only and free - never calls the Claude API. Prefers the digest stored in the
    database and falls back to the digest file on disk. If neither exists, says so and
    names `rebuild_day` (which does cost money) as the way to build one.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    stored = _stored_digest(conn, day)
    if stored is not None:
        return stored

    digest_path = cfg.paths.digests / f"{day}.md"
    if digest_path.is_file():
        return digest_path.read_text(encoding="utf-8")

    return (
        f"No digest found for {day}. Call rebuild_day(day={day!r}) to build one - note "
        "that calls the Claude API and costs money."
    )


@_tool_safe
def list_days(limit: int = 30) -> str:
    """List the most recent local days that have any recorded utterances.

    For each of the last `limit` days (default 30) with at least one utterance, shows
    the local date, how many utterances were recorded, and whether a digest already
    exists for it. Local dates are derived from each utterance's UTC timestamp using the
    configured timezone, so a recording made late at night lands on the correct local
    day even when that crosses a UTC day boundary. Read-only and free.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import to_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    rows = conn.execute("SELECT abs_start_utc FROM utterances").fetchall()
    if not rows:
        return "No utterances recorded yet."

    counts: dict[str, int] = {}
    for row in rows:
        day = to_local(row["abs_start_utc"], tz).strftime("%Y-%m-%d")
        counts[day] = counts.get(day, 0) + 1

    days = sorted(counts, reverse=True)[:limit]
    lines = []
    for day in days:
        digest = "yes" if _stored_digest(conn, day) is not None else "no"
        lines.append(f"{day}: {counts[day]} utterance(s), digest: {digest}")
    return "\n".join(lines)


@_tool_safe
def search_transcript(query: str, day: str = "", limit: int = 30) -> str:
    """Full-text search the local voice transcript for `query`.

    Returns matching lines as `HH:MM [me|other|unknown] text` in local time, oldest
    first. `day` (local date `YYYY-MM-DD`) restricts the search to one day; empty
    searches everything. `limit` caps the number of results (default 30). Read-only and
    free. Any punctuation in `query` (quotes, hyphens, etc.) is treated as literal text,
    never as search syntax.
    """
    from .config import load_config
    from .db import connect
    from .search import search as run_search

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    rows = run_search(conn, query, limit=limit, day=day or None, tz=tz)
    if not rows:
        where = f" on {day}" if day else ""
        return f"No matches for {query!r}{where}."
    return "\n".join(_format_utterance_row(row, tz) for row in rows)


@_tool_safe
def transcript(day: str = "", limit: int = 800) -> str:
    """Return a local day's whole transcript, oldest first.

    `day` is a local date `YYYY-MM-DD`; empty means today. Lines are formatted as
    `HH:MM [me|other|unknown] text`. Truncated to `limit` utterances (default 800) with
    a trailing note when there are more, so a very long day does not blow the response
    up. Read-only and free.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import local_day_bounds, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)

    start, end = local_day_bounds(day, tz)
    rows = conn.execute(
        "SELECT abs_start_utc, speaker, text FROM utterances"
        " WHERE abs_start_utc >= ? AND abs_start_utc < ?"
        " ORDER BY abs_start_utc, t_start_ms",
        (start, end),
    ).fetchall()
    if not rows:
        return f"No utterances recorded for {day}."

    shown = rows[:limit]
    lines = [_format_utterance_row(row, tz) for row in shown]
    if len(rows) > limit:
        lines.append(f"... truncated: showing {limit} of {len(rows)} utterances")
    return "\n".join(lines)


@_tool_safe
def recent(minutes: int = 30, limit: int = 200) -> str:
    """Return the transcript from the last `minutes` minutes, oldest first - this is
    the tool that answers "what was I just talking about" for a live conversation.

    Local, free, read-only - never calls the Claude API. Lines are formatted exactly
    like `transcript`: `HH:MM [me|other|unknown] text` in local time. The header line
    reports the window covered and, separately, how far behind *right now* the newest
    utterance in the database is: with 15-minute file rotation and local ASR decode
    time on top, the freshest transcript can lag the live conversation by several
    minutes even when everything is healthy, and that lag must not be mistaken for
    "nothing was said". `limit` caps the number of utterances returned (default 200).
    """
    from datetime import UTC, datetime, timedelta

    from .config import load_config
    from .db import connect

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    now = datetime.now(UTC)
    cutoff = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = conn.execute(
        "SELECT abs_start_utc, speaker, text FROM utterances"
        " WHERE abs_start_utc >= ?"
        " ORDER BY abs_start_utc, t_start_ms",
        (cutoff,),
    ).fetchall()

    latest = conn.execute("SELECT MAX(abs_start_utc) AS latest FROM utterances").fetchone()[
        "latest"
    ]
    if latest is None:
        lag_note = "no utterances recorded yet"
    else:
        latest_dt = datetime.strptime(latest, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        lag_min = (now - latest_dt).total_seconds() / 60
        lag_note = f"newest utterance in the database is {lag_min:.1f} min old right now"

    header = f"last {minutes} min ({len(rows)} utterance(s)); {lag_note}"
    if not rows:
        return f"{header}\nNo utterances in the last {minutes} minute(s)."

    shown = rows[:limit]
    lines = [header, ""]
    lines.extend(_format_utterance_row(row, tz) for row in shown)
    if len(rows) > limit:
        lines.append(f"... truncated: showing {limit} of {len(rows)} utterances")
    return "\n".join(lines)


def _run_rebuild_day(job: Job, day: str, force: bool) -> str:
    """Background body of `rebuild_day`, run on the job's worker thread.

    `summarize.run_day` (and the `correct.correct_day` it may call first) expose no
    callback or counter for which stage or episode is currently running, and this
    module must not change either of those (another agent owns them right now). So
    the only honest progress available is set once, up front, from what can be
    observed without touching them: how many utterances the day has, and which
    stages `run_day` is about to go through - not a fabricated percentage.
    """
    from .config import load_config
    from .db import connect
    from .summarize import run_day
    from .timeutil import local_day_bounds

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    start_utc, end_utc = local_day_bounds(day, tz)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM utterances WHERE abs_start_utc >= ? AND abs_start_utc < ?",
        (start_utc, end_utc),
    ).fetchone()["n"]
    stages = (
        "a correction pass over the ASR text, then per-episode summarization, then the daily reduce"
        if cfg.correct.enabled
        else "per-episode summarization, then the daily reduce"
    )
    _set_progress(
        job,
        f"{n} utterance(s) for {day}; running {stages}. summarize.run_day exposes no "
        "finer-grained progress than this (no per-episode count, no percentage) - "
        "this is simply still running.",
    )
    return run_day(conn, cfg, day, force=force)


@_tool_safe
def rebuild_day(day: str = "", force: bool = False) -> str:
    """Start rebuilding a local day's summary in the background and return
    IMMEDIATELY with a job id - it does not wait for the rebuild to finish.

    THIS CALLS THE CLAUDE API AND COSTS MONEY (subject to `[llm] daily_budget_usd` in
    config.toml; check `api_usage` for recent spend), and for a day with thousands of
    utterances can take several minutes - well past this session's ~60s tool-call
    timeout. The rebuild (and its API spend) keeps running in the server process even
    after this call returns; nothing about that work is visible or cancellable except
    through `job_status`. Call `job_status(job_id=...)` with the id this returns to
    check progress and, once it finishes, get the resulting Markdown digest (or the
    error, if it failed). Calling `rebuild_day` again for the same `day` while a job
    for it is still running does NOT start a second (paid) job - it returns the
    handle to the one already running instead, so a Claude-Desktop-timeout retry
    cannot double the API spend. If that already-running job is a non-forced rebuild
    and this call asked for `force=True`, the forced recomputation is NOT started
    either (for the same reason: no two concurrent paid rebuilds of one day) - the
    returned message says so explicitly instead of implying the forced work is
    underway, and names the non-forced job's id to wait on before retrying with
    `force=True`. `day` is a local date `YYYY-MM-DD`, default today.
    The Claude correction pass over the day's ASR text runs first whenever
    `[correct] enabled` is set in config.toml, then episodes and the daily digest are
    rebuilt. Summaries already cached for unchanged content are reused unless
    `force=True`, which recomputes the day regardless of caching.
    """
    from .config import load_config
    from .timeutil import today_local

    cfg = load_config()
    day = day or today_local(cfg.summarize.timezone)

    def _work(job: Job) -> str:
        return _run_rebuild_day(job, day, force)

    job, started = _start_job("rebuild_day", day, _work, force=force)
    if not started:
        if force and not job.force:
            # The job `_start_job` handed back is a *different*, non-forced rebuild
            # already in flight for this day - not the forced recompute just asked for.
            # Starting a second, concurrent job for the same day would risk two paid
            # rebuilds running at once, so this call refuses instead - but it must say
            # so plainly rather than reusing the generic "already running" message,
            # which would wrongly imply the requested forced recompute is the one under
            # way (it is not; that job may return a cached digest this call needed
            # bypassed).
            return (
                f"A non-forced rebuild for {day} is already running as job {job.id} "
                f"(elapsed {_elapsed(job)}). The forced recomputation you asked for "
                "has NOT been started - starting a second, concurrent rebuild for the "
                "same day would risk paying for two rebuilds at once. Wait for job "
                f"{job.id} to finish (job_status(job_id={job.id!r})), then call "
                f"rebuild_day(day={day!r}, force=True) again to force the recompute."
            )
        return (
            f"A rebuild for {day} is already running as job {job.id} (elapsed "
            f"{_elapsed(job)}). Not starting a second one - that would double the "
            f"paid Claude API work. Call job_status(job_id={job.id!r}) to check on it."
        )
    return (
        f"Started job {job.id}: rebuilding {day} in the background"
        f"{' with force=True' if force else ''}. This calls the "
        "Claude API and costs money, and keeps running even though this call has "
        f"already returned. Call job_status(job_id={job.id!r}) to check progress and "
        "get the resulting digest (or the error) once it's done."
    )


@_tool_safe
def job_status(job_id: str = "") -> str:
    """Check on a background job started by `rebuild_day` (or list recent ones).

    Read-only and free - never calls the Claude API itself. With `job_id`, reports
    that job's kind and key (e.g. the day, for a rebuild), its state ("running",
    "done", or "failed"), how long it has been running (or took in total), its latest
    progress line, and - once finished - the result (for a rebuild, the digest
    Markdown) or the error message. An unknown `job_id` is reported as such rather
    than raising. With no `job_id`, lists up to the most recent jobs, newest first,
    each with the same state/elapsed-time summary - use this when a job id from an
    earlier, timed-out call was never seen.
    """
    if not job_id:
        with _jobs_lock:
            jobs = sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)
        if not jobs:
            return "No background jobs have been started yet."
        shown = jobs[:_JOB_LIST_LIMIT]
        lines = [f"{len(jobs)} job(s), newest first (showing {len(shown)}):"]
        lines.extend(_format_job_summary(j) for j in shown)
        return "\n".join(lines)

    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return f"No job with id {job_id!r}. Call job_status() with no id to list recent jobs."
    return _format_job_detail(job)


@_tool_safe
def status() -> str:
    """Show `vas` pipeline health: data directory, database path, ASR model, inbox
    file count, recording counts (total, pending, and errored), utterance count, and
    an estimate of how long the pending backlog will take to clear at recent decode
    speed (see `throughput` for the full per-recording breakdown behind that estimate).

    Read-only and free - never touches the network.
    """
    from .config import load_config
    from .db import connect
    from .pipeline import backlog_eta

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    rec = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]
    errored = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE error IS NOT NULL"
    ).fetchone()["n"]
    utt = conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"]
    inbox = 0
    if cfg.paths.inbox.exists():
        inbox = sum(1 for p in cfg.paths.inbox.iterdir() if p.is_file() and p.suffix != ".json")

    return "\n".join(
        [
            f"data_dir : {cfg.paths.root}",
            f"db       : {cfg.paths.db_path}",
            f"asr model: {cfg.asr.model}",
            f"inbox files      : {inbox}",
            f"recordings       : {rec} (pending: {pending}, errored: {errored})",
            f"utterances       : {utt}",
            backlog_eta(conn),
        ]
    )


@_tool_safe
def api_usage(days: int = 30) -> str:
    """Show Claude API token usage and estimated cost recorded by `vas`.

    Aggregates over the last `days` days (default 30) by purpose (map/reduce/correct/
    glossary) and model. Estimated from list prices; only calls made by this tool are
    counted. Read-only and free - it reads a local log, it does not call the API.
    """
    from .config import load_config
    from .llm import USAGE_FILENAME, summarize_usage

    cfg = load_config()
    rows = summarize_usage(cfg.paths.root / USAGE_FILENAME, days=days)
    if not rows:
        return "no API usage recorded"

    lines = [f"{'purpose':<10} {'model':<20} {'calls':>5} {'in':>9} {'out':>8} {'USD':>8}"]
    total = 0.0
    for r in rows:
        total += r["usd"]
        tokens_in = r["input"] + r["cache_read"] + r["cache_write"]
        lines.append(
            f"{r['purpose']:<10} {r['model']:<20} {r['calls']:>5} "
            f"{tokens_in:>9} {r['output']:>8} {r['usd']:>8.3f}"
        )
    lines.append(f"{'total':<10} {'':<20} {'':>5} {'':>9} {'':>8} {total:>8.3f}")
    lines.append("(estimate from list prices; only calls made by vas are counted)")
    return "\n".join(lines)


def _format_realtime_factor(factor: float) -> str:
    """Render `audio_duration / processing_time` unambiguously.

    A factor >= 1 is faster than realtime, rendered so it can never be misread as
    "slower" (e.g. "2.50x faster than realtime", never a bare "2.50x"); below 1 is
    slower than realtime, spelled out the same way.
    """
    if factor >= 1:
        return f"{factor:.2f}x faster than realtime"
    return f"{factor:.2f}x realtime, i.e. slower than realtime"


def _throughput_verdict(factor: float) -> str:
    """One-line plain-language verdict for an aggregate realtime factor."""
    if factor >= 1:
        return (
            f"verdict: {_format_realtime_factor(factor)} - comfortably keeps up with "
            "continuous recording, and can chew through a backlog while still recording."
        )
    return (
        f"verdict: {_format_realtime_factor(factor)} - transcription cannot keep up with "
        "continuous recording; any backlog will only grow until this improves."
    )


@_tool_safe
def throughput(limit: int = 20) -> str:
    """Report whether local ASR transcription is running faster or slower than
    realtime, over the most recently processed recordings.

    Local, free, read-only - reads only the `processing_ms`/`duration_ms` timing
    already stored by `process_recording`; never runs ASR or touches the network. For
    each of the last `limit` processed recordings (default 20) that have both timings
    recorded, reports local start time, source, audio length, processing time, and the
    realtime factor (audio duration / processing time - e.g. "2.50x faster than
    realtime" means 2.5 times faster, never "2.5x slower"). Then reports an aggregate
    realtime factor over the whole sample. When the sample spans more than one ASR
    model (e.g. right after switching faster-whisper on CPU to mlx on the Apple GPU -
    exactly the comparison this is for), the aggregate is broken out per model instead
    of blended together, using the model name recorded on each recording's utterances.
    Ends with a one-line verdict: faster than realtime (and by how much, meaning the
    pipeline can also clear a backlog while it keeps recording) or slower (meaning any
    backlog only grows). Says plainly when there is no timing data yet instead of
    printing a number.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import fmt_hm

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    rows = conn.execute(
        """
        SELECT r.started_at_utc, r.source, r.duration_ms, r.processing_ms,
               (SELECT u.asr_model FROM utterances u
                WHERE u.recording_id = r.id AND u.asr_model IS NOT NULL LIMIT 1) AS asr_model
        FROM recordings r
        WHERE r.processing_ms IS NOT NULL AND r.duration_ms IS NOT NULL
        ORDER BY r.processed_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        return (
            "No timing data yet: no processed recording has both `processing_ms` and "
            "`duration_ms` recorded. Call process_pending to transcribe at least one "
            "recording, then check again."
        )

    lines: list[str] = []
    by_model: dict[str, list[sqlite3.Row]] = {}
    total_audio_ms = 0
    total_proc_ms = 0
    for row in rows:
        audio_ms = row["duration_ms"]
        proc_ms = row["processing_ms"]
        total_audio_ms += audio_ms
        total_proc_ms += proc_ms
        model = row["asr_model"] or "unknown"
        by_model.setdefault(model, []).append(row)
        factor = audio_ms / proc_ms if proc_ms else 0.0
        lines.append(
            f"{fmt_hm(row['started_at_utc'], tz)} [{row['source']}] model={model} "
            f"audio={audio_ms / 1000:.1f}s processing={proc_ms / 1000:.1f}s "
            f"({_format_realtime_factor(factor)})"
        )

    lines.append("")
    if len(by_model) > 1:
        for model, model_rows in by_model.items():
            m_audio_ms = sum(r["duration_ms"] for r in model_rows)
            m_proc_ms = sum(r["processing_ms"] for r in model_rows)
            m_factor = m_audio_ms / m_proc_ms if m_proc_ms else 0.0
            lines.append(
                f"{model}: {len(model_rows)} recording(s), "
                f"{m_audio_ms / 1000:.1f}s of audio, {_format_realtime_factor(m_factor)}"
            )

    overall_factor = total_audio_ms / total_proc_ms if total_proc_ms else 0.0
    lines.append(
        f"overall ({len(rows)} recording(s), {total_audio_ms / 1000:.1f}s of audio): "
        f"{_format_realtime_factor(overall_factor)}"
    )
    lines.append(_throughput_verdict(overall_factor))
    return "\n".join(lines)


@_tool_safe
def list_vocabulary() -> str:
    """Show the personal glossary: proper nouns/aliases/notes and notation style rules
    used to bias ASR, the correction pass, and summarization. Read-only and free.
    """
    from .config import load_config
    from .glossary import load_glossary

    cfg = load_config()
    block = load_glossary(cfg).prompt_block()
    return block if block else "(glossary is empty)"


@_tool_safe
def add_vocabulary(term: str, aliases: list[str] | None = None, note: str = "") -> str:
    """Add a term to the personal glossary, merging into an existing entry if the term
    (or one of its aliases) is already known.

    `term` is the correct spelling of a name, product, project, or piece of jargon.
    `aliases` (optional) are alternate spellings or readings that should resolve to the
    same entry. `note` (optional) is a short description of what/who it is. Affects
    future ASR, correction, and summarization only - it does not rebuild anything
    already generated. Free - no network call.
    """
    from .config import load_config
    from .glossary import Glossary, Term, load_glossary, save_glossary

    cfg = load_config()
    glossary = load_glossary(cfg)
    new_term = Term(term=term, aliases=list(aliases or []), note=note)
    updated = glossary.merge(Glossary(terms=[new_term]))
    save_glossary(cfg, updated)
    return f"added/updated: {term}"


@_tool_safe
def retry_failed() -> str:
    """Re-queue recordings that previously failed to transcribe. Does NOT transcribe them.

    Clears the error flag (recovering a `.part` file left by an in-progress recorder if
    needed) so they become pending again. Local-only, free, and fast - it never runs ASR
    itself, unlike an earlier version of this tool, which ran unbounded local ASR inline
    and could take minutes, well past Claude Desktop's ~60s tool-call timeout. After this,
    call `process_pending` (possibly more than once) to actually transcribe the recordings
    this re-queues.
    """
    from .config import load_config
    from .db import connect
    from .pipeline import retry_failed as run_retry_failed

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    ids = run_retry_failed(conn, cfg)
    if not ids:
        return "no failed recordings"

    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]
    return (
        f"cleared the error flag on {len(ids)} recording(s): {ids}\n"
        f"{pending} recording(s) now pending. Call process_pending to transcribe them."
    )


_backend_cache: dict[tuple[str, str], ASRBackend] = {}


def _cached_backend(cfg: Config) -> ASRBackend:
    """An ASR backend reused across tool calls.

    Claude Desktop keeps this server process alive between calls, and a fresh
    `FasterWhisperBackend` reloads the model on its first transcription -- tens of
    seconds. Clearing a backlog means calling `process_pending` repeatedly, so building
    a new backend each time would spend most of the wall clock loading the same model
    over and over. Keyed by backend and model so a config change still takes effect.
    """
    import os

    from .asr import get_backend

    key = (os.environ.get("VAS_ASR_BACKEND") or cfg.asr.backend, cfg.asr.resolved_model)
    if key not in _backend_cache:
        _backend_cache.clear()
        _backend_cache[key] = get_backend(cfg)
    return _backend_cache[key]


@_tool_safe
def process_pending(limit: int = 3) -> str:
    """Transcribe up to `limit` pending recordings with the local ASR backend.

    Local-only, free - no Claude API call, no network. Each recording takes roughly
    10-60 seconds of CPU/GPU time depending on length and backend, so `limit` defaults
    to a small 3 to stay well inside Claude Desktop's ~60s tool-call timeout: call this
    repeatedly (or pass a larger `limit`) until nothing is pending. This is the tool
    that clears a backlog of pending recordings by hand, e.g. when the launchd worker
    (see `worker_status`) is not running. The very first call may also need to download
    the ASR model, which can be slow - if it seems to hang, that is likely why.
    """
    from .config import load_config
    from .db import connect
    from .pipeline import backlog_eta
    from .pipeline import process_pending as run_process_pending

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    backend = _cached_backend(cfg)
    processed = run_process_pending(conn, cfg, backend, limit=limit)
    remaining = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]

    result = f"processed {processed} recording(s); {remaining} still pending."
    if remaining:
        result += " Call process_pending again (or with a larger limit) to keep clearing it."
    result += "\n" + backlog_eta(conn)
    return result


@_tool_safe
def list_recordings(day: str = "", limit: int = 50) -> str:
    """List each recording ingested on a local day: id, source, local start time,
    duration, utterance count, processing state (processed/pending/errored), the first
    8 characters of its sha256, and its storage path.

    `day` is a local date `YYYY-MM-DD` in the configured timezone; empty means today.
    Sorted by local start time. This is the tool that surfaces two recordings covering
    the same span of time - the shape a `.part`-file duplicate takes (same audio
    ingested twice under different sha256, both transcribed). `find_duplicates` shows
    the effect on transcript text; this shows the underlying recording rows. Read-only,
    local, and free.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import fmt_hm, local_day_bounds, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)
    start, end = local_day_bounds(day, tz)

    rows = conn.execute(
        """
        SELECT r.*, (SELECT COUNT(*) FROM utterances u WHERE u.recording_id = r.id) AS utt_count
        FROM recordings r
        WHERE r.started_at_utc >= ? AND r.started_at_utc < ?
        ORDER BY r.started_at_utc
        """,
        (start, end),
    ).fetchall()
    if not rows:
        return f"No recordings for {day}."

    shown = rows[:limit]
    lines = []
    for row in shown:
        if row["error"] is not None:
            state = "errored"
        elif row["processed_at"] is not None:
            state = "processed"
        else:
            state = "pending"
        duration = f"{row['duration_ms'] / 1000:.1f}s" if row["duration_ms"] is not None else "?"
        lines.append(
            f"id={row['id']} source={row['source']} start={fmt_hm(row['started_at_utc'], tz)} "
            f"duration={duration} utterances={row['utt_count']} state={state} "
            f"sha256={row['sha256'][:8]} path={row['storage_path']}"
        )
    if len(rows) > limit:
        lines.append(f"... truncated: showing {limit} of {len(rows)} recordings")
    return "\n".join(lines)


@_tool_safe
def find_duplicates(day: str = "", limit: int = 40) -> str:
    """Find utterances on a local day whose text repeats, to spot ASR/recorder bugs
    that transcribe the same speech more than once.

    Utterances are grouped by normalized text (surrounding whitespace stripped), and
    two utterances with the same text only count as duplicates of each other when they
    also start within 120 seconds of each other - so a phrase that is genuinely spoken
    twice hours apart is not flagged. For each duplicate group, reports the text once,
    then one line per copy (utterance id, recording id, source, local time). Ends with a
    summary: how many utterances fall in duplicate groups, and which recording-id PAIRS
    co-occur most often across those groups - a pair that dominates means two different
    recordings hold the same audio (e.g. a `.part` file ingested alongside its completed
    re-ingest), while duplicates clustered on a single recording id instead mean that
    one recording was processed more than once and its transcript was appended to
    rather than replaced. `day` is a local date `YYYY-MM-DD`; empty means today. `limit`
    caps how many duplicate groups are printed (default 40). Read-only, local, and
    free.
    """
    from collections import Counter
    from datetime import datetime
    from itertools import combinations

    from .config import load_config
    from .db import connect
    from .timeutil import fmt_hm, local_day_bounds, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)
    start, end = local_day_bounds(day, tz)

    rows = conn.execute(
        """
        SELECT u.id, u.recording_id, u.text, u.abs_start_utc, r.source
        FROM utterances u JOIN recordings r ON r.id = u.recording_id
        WHERE u.abs_start_utc >= ? AND u.abs_start_utc < ?
        ORDER BY u.text, u.abs_start_utc
        """,
        (start, end),
    ).fetchall()
    if not rows:
        return f"No utterances for {day}."

    def _parse(ts: str) -> datetime:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

    clusters: list[list[sqlite3.Row]] = []
    cluster: list[sqlite3.Row] = []
    prev_text: str | None = None
    prev_time: datetime | None = None
    for row in rows:
        text = row["text"].strip()
        t = _parse(row["abs_start_utc"])
        if (
            text == prev_text
            and prev_time is not None
            and abs((t - prev_time).total_seconds()) <= 120
        ):
            cluster.append(row)
        else:
            if cluster:
                clusters.append(cluster)
            cluster = [row]
        prev_text, prev_time = text, t
    if cluster:
        clusters.append(cluster)

    dup_groups = [c for c in clusters if len(c) >= 2]
    if not dup_groups:
        return f"No duplicate utterances found for {day}."

    pair_counter: Counter[tuple[int, int]] = Counter()
    for group in dup_groups:
        ids = sorted(row["recording_id"] for row in group)
        for a, b in combinations(ids, 2):
            pair_counter[(a, b)] += 1

    lines: list[str] = []
    for group in dup_groups[:limit]:
        text = group[0]["text"].strip()
        lines.append(f'"{text}" ({len(group)} copies)')
        for row in group:
            lines.append(
                f"  utt={row['id']} recording={row['recording_id']} source={row['source']} "
                f"time={fmt_hm(row['abs_start_utc'], tz)}"
            )
    if len(dup_groups) > limit:
        lines.append(f"... truncated: showing {limit} of {len(dup_groups)} duplicate group(s)")

    total_dup_utts = sum(len(g) for g in dup_groups)
    lines.append("")
    lines.append(f"summary: {total_dup_utts} utterance(s) in {len(dup_groups)} duplicate group(s)")
    if pair_counter:
        lines.append(
            "most common recording-id pairs (a pair like (5, 5) means one recording's "
            "own utterances duplicated each other, i.e. it was processed more than once):"
        )
        for (a, b), n in pair_counter.most_common(5):
            lines.append(f"  ({a}, {b}): {n} time(s)")
    return "\n".join(lines)


@_tool_safe
def drop_recording(recording_id: int, confirm: bool = False) -> str:
    """Permanently delete one recording's database rows (the recording itself, its
    `segments`, and its `utterances`) - use this to remove a duplicate recording once
    `list_recordings`/`find_duplicates` has identified it.

    Does NOT delete the audio file on disk; only the database rows go away, and the
    output says so. DESTRUCTIVE and irreversible for those rows, so it refuses unless
    `confirm=True` - without it, it prints what WOULD be deleted (source, local start
    time, utterance count, storage path) so the caller can check before confirming.
    Local-only and free.
    """
    from .config import load_config
    from .db import connect
    from .pipeline import delete_recording as run_delete_recording
    from .timeutil import fmt_hm

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    if row is None:
        return f"no such recording: {recording_id}"

    utt_count = conn.execute(
        "SELECT COUNT(*) AS n FROM utterances WHERE recording_id = ?", (recording_id,)
    ).fetchone()["n"]
    description = (
        f"recording {recording_id}: source={row['source']} "
        f"start={fmt_hm(row['started_at_utc'], tz)} utterances={utt_count} "
        f"path={row['storage_path']}"
    )

    if not confirm:
        return (
            f"Would delete {description}\n"
            "The audio file on disk is left untouched either way.\n"
            "Call again with confirm=True to actually delete these database rows."
        )

    run_delete_recording(conn, recording_id)
    return f"deleted {description} (database rows only; audio file left on disk)"


def _repo_dir() -> Path:
    """The git work tree this package is installed from, derived from `__file__`
    (`<repo>/src/voice_ai_summary/mcp_server.py`) rather than the current directory,
    since an MCP server started by Claude Desktop has no meaningful cwd."""
    return Path(__file__).resolve().parents[2]


def _run(cmd: tuple[str, ...], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


@_tool_safe
def update_app() -> str:
    """Pull the latest `vas` code and reinstall it, so updates never need a terminal.

    Runs, in this package's git checkout: `git fetch origin <branch>`, then a
    fast-forward-only merge (refuses and stops cleanly if that is not possible, e.g.
    there are local changes), then `uv pip install -e .` (falling back to
    `python -m pip install -e .` if `uv` is unavailable). The branch is read from the
    `VAS_UPDATE_BRANCH` environment variable, defaulting to `claude/clever-sagan-23tn2w`.
    Refuses if this package is not installed from a git work tree. Restart Claude
    Desktop after a successful update so it picks up the new code.
    """
    import os

    repo_dir = _repo_dir()
    if not (repo_dir / ".git").exists():
        return f"{repo_dir} is not a git work tree; refusing to update."

    branch = os.environ.get("VAS_UPDATE_BRANCH", "claude/clever-sagan-23tn2w")
    output: list[str] = []

    for cmd in (
        ("git", "fetch", "origin", branch),
        ("git", "merge", "--ff-only", f"origin/{branch}"),
    ):
        result = _run(cmd, repo_dir, timeout=60)
        output.append(f"$ {' '.join(cmd)}\n{(result.stdout + result.stderr).strip()}")
        if result.returncode != 0:
            output.append(
                "Update stopped: the step above failed (a fast-forward merge needs no "
                "local changes on this branch). Resolve it in a terminal, then retry."
            )
            return "\n\n".join(output)

    install_cmd: tuple[str, ...] = ("uv", "pip", "install", "-e", ".")
    try:
        result = _run(install_cmd, repo_dir, timeout=300)
    except FileNotFoundError:
        result = None
    if result is None or result.returncode != 0:
        install_cmd = (sys.executable, "-m", "pip", "install", "-e", ".")
        result = _run(install_cmd, repo_dir, timeout=300)

    output.append(f"$ {' '.join(install_cmd)}\n{(result.stdout + result.stderr).strip()}")
    if result.returncode != 0:
        output.append("Install failed; see output above.")
        return "\n\n".join(output)

    output.append("Update complete. Restart Claude Desktop to use the new code.")
    return "\n\n".join(output)


def _agents_dir() -> Path:
    """Where launchd plist files for this app live, mirroring `launchd.install`."""
    return Path.home() / "Library" / "LaunchAgents"


_LAUNCHCTL_PRINT_INTERESTING_KEYS = frozenset(
    {"state", "pid", "last exit status", "last exit reason"}
)


def _launchctl_print_summary(output: str) -> list[str]:
    """Pull the lines worth reporting out of `launchctl print`'s (long, indented) output:
    run state, pid, and the last exit status/reason - not the whole dump."""
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip().lower()
        if key in _LAUNCHCTL_PRINT_INTERESTING_KEYS:
            lines.append(stripped)
    return lines


def _report_one_service(label: str) -> str:
    """One label's worth of `worker_status` output: plist presence plus what launchd
    itself reports, preferring `launchctl print` and falling back to `launchctl list`."""
    import os

    plist_path = _agents_dir() / f"{label}.plist"
    lines = [
        f"{label}:",
        f"  plist: {'present' if plist_path.exists() else 'MISSING'} ({plist_path})",
    ]

    uid = os.getuid()
    try:
        printed = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"  launchctl print failed to run: {exc}")
        printed = None

    if printed is not None and printed.returncode == 0:
        summary = _launchctl_print_summary(printed.stdout)
        if summary:
            lines.extend(f"  {line}" for line in summary)
        else:
            lines.append("  loaded, but no state/pid/exit lines found in launchctl print")
        return "\n".join(lines)

    # `launchctl print` fails (typically exit 113/1) when the service is not
    # bootstrapped into this session's domain; fall back to `launchctl list`.
    try:
        listed = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"  launchctl list failed to run too: {exc}")
        return "\n".join(lines)

    if listed.returncode == 0 and listed.stdout.strip():
        lines.append(f"  not found via launchctl print; launchctl list: {listed.stdout.strip()}")
    else:
        lines.append("  not loaded (launchctl list has no entry for this label)")
    return "\n".join(lines)


@_tool_safe
def worker_status() -> str:
    """Report what launchd actually knows about the background worker and digest
    services - this is the tool that answers "why isn't anything being transcribed".

    For both the worker (transcribes recordings continuously) and the nightly digest
    job, reports whether `~/Library/LaunchAgents/<label>.plist` exists and, via
    `launchctl print` (falling back to `launchctl list`), whether the service is
    currently loaded, its run state, pid, and last exit status/reason. Read-only,
    local, and free - macOS only; on any other platform it says so plainly instead of
    guessing. If the worker looks stopped or crash-looping, `restart_worker` fixes it,
    `install_services` reinstalls it if the plist itself looks stale, and
    `service_logs` shows why it has been failing.
    """
    if sys.platform != "darwin":
        return "launchd is macOS-only; there is nothing to report on this platform."

    from . import launchd

    return "\n\n".join(
        _report_one_service(label) for label in (launchd.WORKER_LABEL, launchd.DIGEST_LABEL)
    )


@_tool_safe
def restart_worker() -> str:
    """Restart the background transcription worker via launchd - use this when
    `worker_status` shows it stopped, crashed, or otherwise not running.

    Local-only, free, and fast (a few seconds). Tries `launchctl kickstart -k` first
    (restarts an already-bootstrapped service in place); if that fails - typically
    because the service was never bootstrapped into this login session - falls back to
    `launchctl bootout` then `launchctl bootstrap` from the installed plist. Refuses
    cleanly, naming `install_services`, if the worker's plist does not exist at all.
    macOS only.
    """
    if sys.platform != "darwin":
        return "launchd is macOS-only; there is nothing to restart on this platform."

    import os

    from . import launchd

    plist_path = _agents_dir() / f"{launchd.WORKER_LABEL}.plist"
    if not plist_path.exists():
        return (
            f"{plist_path} does not exist - the worker has never been installed. "
            "Call install_services() first."
        )

    uid = os.getuid()
    try:
        kickstarted = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{launchd.WORKER_LABEL}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"launchctl kickstart failed to run: {exc}"

    if kickstarted.returncode == 0:
        detail = (kickstarted.stdout + kickstarted.stderr).strip()
        return f"restarted {launchd.WORKER_LABEL} via launchctl kickstart.\n{detail}"

    output = [
        f"launchctl kickstart failed (exit {kickstarted.returncode}): "
        f"{(kickstarted.stdout + kickstarted.stderr).strip()}",
        "falling back to bootout + bootstrap ...",
    ]
    launchd.launchctl_bootout(uid, str(plist_path))
    try:
        launchd.launchctl_bootstrap(uid, str(plist_path))
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        output.append(f"bootstrap failed: {detail or exc}")
        return "\n".join(output)
    except subprocess.TimeoutExpired as exc:
        output.append(f"bootstrap timed out: {exc}")
        return "\n".join(output)

    output.append(f"bootout + bootstrap succeeded for {launchd.WORKER_LABEL}.")
    return "\n".join(output)


@_tool_safe
def install_services() -> str:
    """(Re)install the launchd worker and digest services from current config.toml.

    This is how to fix a stale or wrong LaunchAgent - e.g. after editing config.toml,
    moving the `vas` binary, or if `worker_status` shows the plist is missing or the
    service is misbehaving. Writes both plist files to ~/Library/LaunchAgents and, on
    macOS, boots each service out and re-bootstraps it so launchd picks up the new
    file. Safe to run repeatedly - it is the same install every `vas install-launchd`
    in a terminal does. Local-only and free; not available on non-macOS beyond writing
    the plist files.
    """
    from . import launchd
    from .config import load_config

    cfg = load_config()
    paths = launchd.install(cfg)
    listing = "\n".join(f"  {p}" for p in paths)

    if sys.platform == "darwin":
        note = (
            "Booted each service out and re-bootstrapped it, so launchd is running the new plist."
        )
    else:
        note = "launchd is macOS-only, so only the plist files were written (no launchctl call)."
    return f"Wrote:\n{listing}\n{note}"


@_tool_safe
def service_logs(service: str = "worker", lines: int = 40) -> str:
    """Tail the launchd stdout/stderr log files for the worker or digest service.

    `service` is `"worker"` or `"digest"` (anything else is rejected); `lines` caps how
    many trailing lines of each file are shown (default 40). Reads
    `~/Library/Logs/VoiceAISummary/com.voiceaisummary.<service>.log` (stdout) and `.err`
    (stderr), the same files `install_services` configures launchd to write to. Says
    plainly when a file is missing (service never ran, or was installed before logging
    existed) or empty. Read-only and free - this is usually the next step after
    `worker_status` shows a service crashing or exiting with an error.
    """
    from . import launchd

    labels = {"worker": launchd.WORKER_LABEL, "digest": launchd.DIGEST_LABEL}
    if service not in labels:
        return f"unknown service {service!r}; expected 'worker' or 'digest'."

    label = labels[service]
    directory = launchd.log_dir()
    out: list[str] = []
    for suffix, kind in ((".log", "stdout"), (".err", "stderr")):
        path = directory / f"{label}{suffix}"
        out.append(f"--- {kind}: {path} ---")
        if not path.exists():
            out.append("(file does not exist - the service may never have run)")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            out.append("(empty)")
            continue
        out.extend(text.splitlines()[-lines:])
    return "\n".join(out)


for _fn in (
    daily_summary,
    list_days,
    search_transcript,
    transcript,
    recent,
    rebuild_day,
    job_status,
    status,
    api_usage,
    throughput,
    list_vocabulary,
    add_vocabulary,
    retry_failed,
    process_pending,
    list_recordings,
    find_duplicates,
    drop_recording,
    worker_status,
    restart_worker,
    install_services,
    service_logs,
    update_app,
):
    server.add_tool(_fn)
del _fn


def main() -> None:
    """Entry point for the `vas-mcp` console script: run the stdio MCP server."""
    server.run("stdio")


if __name__ == "__main__":
    main()
