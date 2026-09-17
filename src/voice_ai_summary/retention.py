"""Retention: prune recorded audio and append-only logs so `store/` doesn't grow
forever. Transcripts and digests are never touched here - only the audio file behind a
recording (once it is old enough and either transcribed or permanently failed) and the
append-only `.jsonl` logs / launchd stdout+stderr logs.

Nothing here deletes a whole recording's row, its transcript, or a digest - that is
`pipeline.delete_range`'s job, for a caller who explicitly asked to erase a time range
of recorded speech. This module only ever removes what `RetentionConfig` says can be
regenerated or is no longer needed: audio (the transcript survives it), old log lines,
oversized launchd logs, and directories left empty by the audio deletions above.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import launchd
from .config import Config
from .db import utcnow_iso
from .llm import USAGE_FILENAME
from .recorder_state import EVENTS_FILENAME
from .security import AUDIT_FILENAME, open_private

logger = logging.getLogger(__name__)

# The launchd worker log basenames, for `run_worker` to pass as `skip_logs`: truncating
# the very file the worker process is itself appending stdout/stderr to is not safe (see
# `_truncate_log_tail`'s docstring for why an in-place rewrite is used at all, and why
# even that is skipped for a file this same process still has open for writing).
WORKER_LOG_BASENAMES = frozenset({f"{launchd.WORKER_LABEL}.log", f"{launchd.WORKER_LABEL}.err"})


@dataclass
class PruneReport:
    audio_deleted: int = 0
    audio_bytes: int = 0
    # A row said its audio was still on disk (no `audio_deleted_at` yet) but the file was
    # already gone - counted here rather than as an error, and the row is still stamped.
    audio_missing: int = 0
    # A unlink that failed for a reason other than "already missing" (e.g. permissions).
    # Logged, counted, and skipped - the row is left unstamped so it is retried next time.
    audio_skipped_errors: int = 0
    usage_lines_dropped: int = 0
    events_lines_dropped: int = 0
    audit_lines_dropped: int = 0
    logs_truncated: list[str] = field(default_factory=list)
    empty_dirs_removed: int = 0
    dry_run: bool = True

    def summary(self) -> str:
        prefix = "[dry run] " if self.dry_run else ""
        lines = [
            f"{prefix}audio: {self.audio_deleted} file(s) deleted ({self.audio_bytes} bytes), "
            f"{self.audio_missing} already missing, "
            f"{self.audio_skipped_errors} skipped after a failed delete",
            f"{prefix}logs: usage {self.usage_lines_dropped} line(s) dropped, "
            f"recorder events {self.events_lines_dropped} line(s) dropped, "
            f"audit {self.audit_lines_dropped} line(s) dropped",
            f"{prefix}launchd logs truncated: {len(self.logs_truncated)}"
            + (f" ({', '.join(self.logs_truncated)})" if self.logs_truncated else ""),
            f"{prefix}empty store directories removed: {self.empty_dirs_removed}",
        ]
        return "\n".join(lines)


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cutoff(now: datetime, days: int) -> str | None:
    """The ISO cutoff `days` before `now`, or `None` if the rule is disabled (`days<=0`)."""
    if days <= 0:
        return None
    return _fmt(now - timedelta(days=days))


def _prune_one_audio(conn, cfg: Config, row, report: PruneReport, *, dry_run: bool) -> None:
    path = cfg.paths.store / row["storage_path"]
    if dry_run:
        try:
            size = path.stat().st_size
        except OSError:
            report.audio_missing += 1
            return
        report.audio_deleted += 1
        report.audio_bytes += size
        return

    try:
        size = path.stat().st_size
        path.unlink()
    except FileNotFoundError:
        report.audio_missing += 1
    except OSError as exc:
        logger.warning("failed to delete audio for recording %s: %s", row["id"], exc)
        report.audio_skipped_errors += 1
        return
    else:
        report.audio_deleted += 1
        report.audio_bytes += size

    # Reached for both a successful unlink and a missing file - either way the audio is
    # gone, so the row is stamped. Only the genuine "failed to unlink" case above skips
    # this and leaves the row eligible for another attempt.
    conn.execute(
        "UPDATE recordings SET audio_deleted_at = ? WHERE id = ?", (utcnow_iso(), row["id"])
    )
    conn.commit()


def _prune_audio(conn, cfg: Config, now: datetime, report: PruneReport, *, dry_run: bool) -> None:
    audio_cutoff = _cutoff(now, cfg.retention.audio_days)
    errored_cutoff = _cutoff(now, cfg.retention.errored_audio_days)
    rows = conn.execute(
        "SELECT id, storage_path, processed_at, error, ingested_at FROM recordings"
        " WHERE audio_deleted_at IS NULL AND storage_path IS NOT NULL"
    ).fetchall()
    for row in rows:
        if row["error"] is not None:
            cutoff, marker = errored_cutoff, row["ingested_at"]
        elif row["processed_at"] is not None:
            cutoff, marker = audio_cutoff, row["processed_at"]
        else:
            continue  # pending (never transcribed, never errored): never delete its audio
        if cutoff is None or marker is None or marker >= cutoff:
            continue
        _prune_one_audio(conn, cfg, row, report, dry_run=dry_run)


def _line_timestamp(line: str) -> datetime | None:
    """Best-effort timestamp for one jsonl line, or `None` if it can't be read - a line
    with no parsable timestamp is always kept, never guessed away."""
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(rec, dict):
        return None
    for key in ("ts", "updated_at", "at"):
        value = rec.get(key)
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
            except ValueError:
                continue
    return None


def _prune_jsonl(path: Path, days: int, now: datetime, *, dry_run: bool) -> int:
    """Drop lines older than `days` from `path`. Returns the number of lines dropped.

    A line whose timestamp can't be parsed is always kept. Rewritten through a temp file
    opened with `security.open_private` (mode 0600) then `os.replace`d over the original,
    so a log that predates this module gets locked down the first time it is pruned.
    """
    if days <= 0 or not path.is_file():
        return 0
    cutoff = now - timedelta(days=days)
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        ts = _line_timestamp(line)
        if ts is not None and ts < cutoff:
            dropped += 1
            continue
        kept.append(line)
    if dropped == 0 or dry_run:
        return dropped

    tmp_path = path.with_name(path.name + ".tmp")
    with open_private(tmp_path, "w") as f:
        for line in kept:
            f.write(line + "\n")
    os.replace(tmp_path, path)
    return dropped


def _truncate_log_tail(path: Path, max_bytes: int) -> None:
    """Cut `path` back to (roughly) its last `max_bytes // 2` bytes, aligned to a line
    boundary, in place.

    launchd holds this file open for append (O_APPEND) for as long as the worker/digest
    service runs. Rewriting "from the front" the normal way - write a new file, then
    rename it over the old one - would leave launchd's fd pointing at the old, now
    unlinked inode; every subsequent line the service writes would vanish into a file
    nothing reads any more. Rewriting in place (seek, write, truncate on the same open
    fd) keeps the same inode, so launchd's fd - and its append position - stays valid.
    """
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size <= max_bytes:
            return
        keep = max_bytes // 2
        f.seek(max(size - keep, 0))
        tail = f.read()
        newline = tail.find(b"\n")
        if newline != -1:
            tail = tail[newline + 1 :]
        f.seek(0)
        f.write(tail)
        f.truncate()


def _prune_launchd_logs(
    log_dir: Path, max_bytes: int, *, skip_logs: frozenset[str] | set[str], dry_run: bool
) -> list[str]:
    if max_bytes <= 0 or not log_dir.is_dir():
        return []
    truncated: list[str] = []
    candidates = sorted(log_dir.glob("*.log")) + sorted(log_dir.glob("*.err"))
    for path in candidates:
        if path.name in skip_logs or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= max_bytes:
            continue
        if not dry_run:
            try:
                _truncate_log_tail(path, max_bytes)
            except OSError:
                logger.exception("failed to truncate log %s", path)
                continue
        truncated.append(path.name)
    return truncated


def _remove_empty_dirs(store: Path, *, dry_run: bool) -> int:
    """Remove now-empty `store/YYYY/MM/DD` (and `MM`, `YYYY`) directories, bottom-up."""
    if not store.is_dir():
        return 0
    removed = 0
    for year_dir in sorted(p for p in store.iterdir() if p.is_dir()):
        for month_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            for day_dir in sorted(p for p in month_dir.iterdir() if p.is_dir()):
                if not any(day_dir.iterdir()):
                    removed += 1
                    if not dry_run:
                        day_dir.rmdir()
            if not any(month_dir.iterdir()):
                removed += 1
                if not dry_run:
                    month_dir.rmdir()
        if not any(year_dir.iterdir()):
            removed += 1
            if not dry_run:
                year_dir.rmdir()
    return removed


def prune(
    conn,
    cfg: Config,
    *,
    now: datetime | None = None,
    dry_run: bool = True,
    log_dir: Path | None = None,
    skip_logs: set[str] = frozenset(),
) -> PruneReport:
    """Apply every retention rule in `cfg.retention` once, and report what happened.

    `dry_run=True` (the default) computes the report without touching disk or the
    database - a preview safe to call any time. `dry_run=False` actually deletes audio,
    rewrites the jsonl logs, truncates oversized launchd logs, and removes directories
    the audio deletions left empty.
    """
    now = now if now is not None else datetime.now(UTC)
    report = PruneReport(dry_run=dry_run)

    _prune_audio(conn, cfg, now, report, dry_run=dry_run)

    root = cfg.paths.root
    report.usage_lines_dropped = _prune_jsonl(
        root / USAGE_FILENAME, cfg.retention.usage_log_days, now, dry_run=dry_run
    )
    report.events_lines_dropped = _prune_jsonl(
        root / EVENTS_FILENAME, cfg.retention.recorder_events_days, now, dry_run=dry_run
    )
    report.audit_lines_dropped = _prune_jsonl(
        root / AUDIT_FILENAME, cfg.retention.audit_log_days, now, dry_run=dry_run
    )

    if log_dir is not None:
        report.logs_truncated = _prune_launchd_logs(
            log_dir, cfg.retention.log_max_bytes, skip_logs=skip_logs, dry_run=dry_run
        )

    report.empty_dirs_removed = _remove_empty_dirs(cfg.paths.store, dry_run=dry_run)

    return report
