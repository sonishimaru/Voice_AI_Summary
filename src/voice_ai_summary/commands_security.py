"""CLI subcommands for retention and range deletion. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app


@app.command("prune")
def prune_cmd(
    yes: bool = typer.Option(
        False, "--yes", help="Actually delete/rewrite; without it, only preview."
    ),
) -> None:
    """Apply retention (old audio, log rotation) once. Previews by default."""
    from .config import load_config
    from .db import connect
    from .launchd import log_dir
    from .retention import WORKER_LOG_BASENAMES, prune

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    report = prune(
        conn,
        cfg,
        dry_run=not yes,
        log_dir=log_dir(),
        skip_logs=WORKER_LOG_BASENAMES,
    )
    typer.echo(report.summary())
    if not yes:
        typer.echo("\n(dry run - pass --yes to actually delete/rewrite)")


@app.command("delete-range")
def delete_range_cmd(
    day: str = typer.Option(..., "--day", help="Local date YYYY-MM-DD the range falls in."),
    start: str = typer.Option(..., "--start", help="Local start time HH:MM (24:00 allowed)."),
    end: str = typer.Option(..., "--end", help="Local end time HH:MM (24:00 allowed)."),
    yes: bool = typer.Option(
        False, "--yes", help="Actually delete; without it, only preview what would go."
    ),
) -> None:
    """Permanently delete every recording overlapping a local time range.

    Deletion is per-recording: the whole recording(s) covering any part of the requested
    range are removed, not just the requested minutes - see the preview before passing
    --yes. Also removes the affected days' cached summaries and digest files; rebuild
    them afterwards with `vas digest` (or the `rebuild_day` MCP tool).
    """
    from .config import load_config
    from .db import connect
    from .pipeline import delete_range
    from .timeutil import local_time_to_utc

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    try:
        start_utc = local_time_to_utc(day, start, tz)
        end_utc = local_time_to_utc(day, end, tz)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if end_utc <= start_utc:
        raise typer.BadParameter("--end must be after --start")

    report = delete_range(conn, cfg, start_utc, end_utc, dry_run=not yes)
    typer.echo(report.summary(tz))
    if not yes:
        typer.echo("\n(dry run - pass --yes to actually delete)")
