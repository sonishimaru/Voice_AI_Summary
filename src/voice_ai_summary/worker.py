"""Background worker: poll the inbox, ingest, transcribe, repeat."""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from .asr import get_backend
from .config import Config
from .db import connect
from .ingest import ingest_inbox
from .pipeline import process_pending

logger = logging.getLogger(__name__)


class _Stop(BaseException):
    """Shutdown request from SIGINT/SIGTERM.

    Deliberately a `BaseException`: it is raised from a signal handler, so it can land
    anywhere -- including inside `process_recording`, whose `except Exception` would
    otherwise record the shutdown as a transcription failure and carry on, ignoring the
    signal. `launchctl kickstart -k` does exactly that on every worker restart.
    """


def run_worker(cfg: Config, *, poll_seconds: int | None = None, once: bool = False) -> None:
    """Ingest + process in a loop until interrupted (or once, if `once` is set)."""
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    backend = get_backend(cfg)
    interval = poll_seconds if poll_seconds is not None else cfg.schedule.worker_poll_seconds

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
            processed = process_pending(conn, cfg, backend)
            if processed:
                logger.info("processed %d recording(s)", processed)
            if once:
                return
            time.sleep(interval)
    except _Stop:
        logger.info("worker interrupted, stopping")
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
