"""Background worker: poll the inbox, ingest, transcribe, repeat."""

from __future__ import annotations

import logging
import signal
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from . import retention
from .asr import get_backend
from .config import Config
from .db import connect
from .deliver.notify import send_desktop_notification
from .ingest import ingest_inbox
from .launchd import log_dir as launchd_log_dir
from .pipeline import process_pending
from .recorder_state import RecorderState, is_alive, read_state

logger = logging.getLogger(__name__)

# How often `run_worker` calls `retention.prune` - once an hour is plenty for something
# that only deletes audio once it is already `audio_days`/`errored_audio_days` old and
# rewrites append-only logs; there's no benefit to running it on every poll.
PRUNE_INTERVAL_S = 60 * 60


class _Stop(BaseException):
    """Shutdown request from SIGINT/SIGTERM.

    Deliberately a `BaseException`: it is raised from a signal handler, so it can land
    anywhere -- including inside `process_recording`, whose `except Exception` would
    otherwise record the shutdown as a transcription failure and carry on, ignoring the
    signal. `launchctl kickstart -k` does exactly that on every worker restart.
    """


def _pending_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()[0]


class _BacklogWatch:
    """Tracks whether the pending-recording count has stayed at/above `count_threshold`
    for at least `grace_s` without dropping back below it, and fires at most one
    notification per such episode.

    Armed again only once the backlog actually recovers (drops below the threshold) -
    a worker that is merely slow, oscillating around the threshold without recovering,
    must not get re-notified on every poll while still backlogged.
    """

    def __init__(
        self,
        *,
        count_threshold: int,
        grace_s: float,
        clock: Callable[[], float],
        notify: Callable[[int, float], None],
    ) -> None:
        self._count_threshold = count_threshold
        self._grace_s = grace_s
        self._clock = clock
        self._notify = notify
        self._backlog_since: float | None = None
        self._fired = False

    def observe(self, pending: int) -> None:
        if self._count_threshold <= 0 or self._grace_s <= 0:
            return
        if pending < self._count_threshold:
            self._backlog_since = None
            self._fired = False
            return
        now = self._clock()
        if self._backlog_since is None:
            self._backlog_since = now
            return
        elapsed = now - self._backlog_since
        if not self._fired and elapsed >= self._grace_s:
            self._fired = True
            try:
                self._notify(pending, elapsed)
            except Exception:
                logger.exception("failed to send backlog alert notification")


def _notify_backlog(pending: int, elapsed_s: float) -> None:
    minutes = int(elapsed_s // 60)
    send_desktop_notification(
        f"{pending} 件の録音が {minutes} 分以上処理されずに残っています。"
        "worker が動作しているか確認してください。",
        title="文字起こしが滞留しています",
    )


class _RecorderWatch:
    """Tracks whether the recorder app reports `recording` (and looks alive) while no
    inbox file has arrived in at least `3 * rotation_s`, and fires at most one
    notification per such episode.

    The recorder rotates a fresh file into the inbox roughly every `rotation_s`
    seconds while actually recording, so three missed rotations in a row means
    audio has stopped flowing even though the app still claims to be recording -
    the app crashed mid-recording, lost mic/system audio permission, or is stuck.

    Silent (and reset) whenever the state file says paused/stopped, is missing, or
    the process looks dead - none of those mean audio should be arriving. Armed
    again as soon as a file arrives.
    """

    def __init__(
        self,
        *,
        rotation_s: float,
        clock: Callable[[], float],
        notify: Callable[[float], None],
        read_state: Callable[[], RecorderState | None],
        is_alive: Callable[[RecorderState], bool],
    ) -> None:
        self._rotation_s = rotation_s
        self._clock = clock
        self._notify = notify
        self._read_state = read_state
        self._is_alive = is_alive
        self._last_arrival: float | None = None
        self._fired = False

    def observe(self, now_files_seen: bool) -> None:
        state = self._read_state()
        if state is None or state.state != "recording" or not self._is_alive(state):
            self._last_arrival = None
            self._fired = False
            return

        now = self._clock()
        if now_files_seen:
            self._last_arrival = now
            self._fired = False
            return
        if self._last_arrival is None:
            # First poll that sees `recording` with nothing ingested yet - start the
            # clock now rather than assuming a file is already overdue.
            self._last_arrival = now
            return

        elapsed = now - self._last_arrival
        if not self._fired and elapsed >= 3 * self._rotation_s:
            self._fired = True
            try:
                self._notify(elapsed)
            except Exception:
                logger.exception("failed to send recorder-health alert notification")


def _notify_recorder_health(elapsed_s: float) -> None:
    minutes = int(elapsed_s // 60)
    send_desktop_notification(
        f"録音アプリは録音中と報告していますが {minutes} 分間ファイルが届いていません。",
        title="録音ファイルが届いていません",
    )


def run_worker(
    cfg: Config,
    *,
    poll_seconds: int | None = None,
    once: bool = False,
    clock: Callable[[], float] | None = None,
) -> None:
    """Ingest + process in a loop until interrupted (or once, if `once` is set).

    Does not set the process umask itself - `cli.main()` (the `vas worker` entry point)
    already does that before dispatching here, so the worker just inherits it. Setting
    it again from inside `run_worker` would also mean a test calling it directly (there
    is no other entry point that reaches this function without going through
    `cli.main()`) changes this whole test process's umask as a side effect, which
    outlives the test.
    """
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    backend = get_backend(cfg)
    interval = poll_seconds if poll_seconds is not None else cfg.schedule.worker_poll_seconds
    clock_fn = clock if clock is not None else time.monotonic

    backlog_watch = _BacklogWatch(
        count_threshold=cfg.schedule.backlog_alert_count,
        grace_s=cfg.schedule.backlog_alert_minutes * 60,
        clock=clock_fn,
        notify=_notify_backlog,
    )
    recorder_watch = _RecorderWatch(
        rotation_s=cfg.recorder.rotation_minutes * 60,
        clock=clock_fn,
        notify=_notify_recorder_health,
        read_state=lambda: read_state(cfg.paths.root),
        is_alive=lambda state: is_alive(state, now=datetime.now(UTC)),
    )

    # `None` means "never pruned yet this run" - the first poll always prunes, then no
    # more often than once per `PRUNE_INTERVAL_S` after that.
    last_prune: float | None = None

    def _handle_signal(signum: int, _frame: Any) -> None:
        raise _Stop

    old_handlers = {
        sig: signal.signal(sig, _handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        while True:
            ingested = ingest_inbox(conn, cfg)
            if ingested:
                logger.info("ingested %d recording(s)", len(ingested))
            try:
                recorder_watch.observe(bool(ingested))
            except Exception:
                logger.exception("recorder-health watch failed")
            processed = process_pending(conn, cfg, backend)
            if processed:
                logger.info("processed %d recording(s)", processed)
            backlog_watch.observe(_pending_count(conn))

            now = clock_fn()
            if last_prune is None or now - last_prune >= PRUNE_INTERVAL_S:
                last_prune = now
                try:
                    retention.prune(
                        conn,
                        cfg,
                        dry_run=False,
                        log_dir=launchd_log_dir(),
                        skip_logs=retention.WORKER_LOG_BASENAMES,
                    )
                except Exception:
                    logger.exception("retention prune failed")

            if once:
                return
            time.sleep(interval)
    except _Stop:
        logger.info("worker interrupted, stopping")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
