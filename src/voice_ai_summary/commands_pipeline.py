"""CLI subcommands for the pipeline stage. Registered on `app` at import time."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli import app


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Audio files to ingest."),  # noqa: B008
    source: str | None = typer.Option(
        None, "--source", help="Override source (mac_mic/mac_system/file)."
    ),
    started_at: str | None = typer.Option(
        None, "--started-at", help="Override started_at_utc, e.g. 2026-09-15T01:02:03Z."
    ),
    tz: str = typer.Option("+00:00", "--tz", help="Timezone offset, e.g. +09:00."),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving the source file."),
) -> None:
    """Ingest one or more recording files into the store."""
    from .config import load_config
    from .db import connect
    from .ingest import ingest_file

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    for path in paths:
        rec_id = ingest_file(
            conn,
            cfg,
            path,
            source=source,
            started_at_utc=started_at,
            tz_offset=tz,
            move=not copy,
        )
        typer.echo(f"{path} -> recording {rec_id}")


@app.command()
def process(
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of recordings to process."
    ),
) -> None:
    """Transcribe all pending recordings."""
    from .asr import get_backend
    from .config import load_config
    from .db import connect
    from .pipeline import process_pending

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    backend = get_backend(cfg)
    count = process_pending(conn, cfg, backend, limit=limit)
    typer.echo(f"processed {count} recording(s)")


@app.command()
def retry() -> None:
    """Re-queue recordings that failed to process, then process them."""
    from .asr import get_backend
    from .config import load_config
    from .db import connect
    from .pipeline import process_pending, retry_failed

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    ids = retry_failed(conn, cfg)
    if not ids:
        typer.echo("no failed recordings")
        return
    typer.echo(f"re-queued {len(ids)} recording(s): {ids}")
    count = process_pending(conn, cfg, get_backend(cfg))
    typer.echo(f"processed {count} recording(s)")


@app.command()
def reprocess(
    day: str | None = typer.Option(None, "--day", help="Local date YYYY-MM-DD to re-transcribe."),
    all_recordings: bool = typer.Option(False, "--all", help="Re-transcribe every recording."),
    no_process: bool = typer.Option(False, "--no-process", help="Only reset; don't transcribe."),
) -> None:
    """Re-transcribe recordings with the current ASR/VAD settings."""
    from .asr import get_backend
    from .config import load_config
    from .db import connect
    from .pipeline import process_pending, reset_recordings
    from .timeutil import local_day_bounds

    if bool(day) == all_recordings:
        raise typer.BadParameter("pass exactly one of --day or --all")
    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    if all_recordings:
        rows = conn.execute("SELECT id FROM recordings").fetchall()
    else:
        start, end = local_day_bounds(day, cfg.summarize.timezone)
        rows = conn.execute(
            "SELECT id FROM recordings WHERE started_at_utc >= ? AND started_at_utc < ?",
            (start, end),
        ).fetchall()
    ids = [r["id"] for r in rows]
    typer.echo(f"reset {reset_recordings(conn, ids)} recording(s)")
    if ids and not no_process:
        count = process_pending(conn, cfg, get_backend(cfg))
        typer.echo(f"processed {count} recording(s)")


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Run a single ingest+process pass and exit."),
) -> None:
    """Run the ingest/transcribe loop."""
    from .config import load_config
    from .worker import run_worker

    cfg = load_config()
    cfg.ensure_dirs()
    run_worker(cfg, once=once)


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to search for."),
    limit: int = typer.Option(50, "--limit", help="Maximum number of results."),
    day: str | None = typer.Option(None, "--day", help="Restrict to a local date, YYYY-MM-DD."),
) -> None:
    """Search transcribed utterances."""
    from .config import load_config
    from .db import connect
    from .search import search as run_search

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    for row in run_search(conn, query, limit=limit, day=day):
        ts = row["abs_start_utc"][11:19]
        typer.echo(f"{ts} [{row['speaker']}] {row['text']}")
