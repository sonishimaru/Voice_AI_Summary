"""CLI subcommands for the deliver stage. Registered on `app` at import time."""

from __future__ import annotations

import sqlite3
import time

import typer

from .cli import app
from .config import Config, load_config
from .db import connect

# How many recordings `_catch_up_pending` asks `pipeline.process_pending` to transcribe
# per batch. Small so the elapsed-budget check between batches actually bites, instead
# of one call draining an arbitrarily large backlog before the budget is next checked.
_CATCHUP_BATCH = 5


def _pending_and_errored_counts(
    conn: sqlite3.Connection, start_utc: str, end_utc: str
) -> tuple[int, int]:
    """(pending, errored) recording counts for the local day covering `[start_utc, end_utc)`."""
    pending = conn.execute(
        "SELECT COUNT(*) FROM recordings WHERE started_at_utc >= ? AND started_at_utc < ?"
        " AND processed_at IS NULL AND error IS NULL",
        (start_utc, end_utc),
    ).fetchone()[0]
    errored = conn.execute(
        "SELECT COUNT(*) FROM recordings WHERE started_at_utc >= ? AND started_at_utc < ?"
        " AND error IS NOT NULL",
        (start_utc, end_utc),
    ).fetchone()[0]
    return pending, errored


def _catch_up_pending(conn: sqlite3.Connection, cfg: Config, budget_s: int) -> None:
    """Transcribe globally-pending recordings (oldest first) in small batches, stopping
    once `budget_s` wall-clock seconds have elapsed.

    Delegates the actual transcription to `pipeline.process_pending` - this just bounds
    how much of it a single `vas digest` run is willing to wait for, so a dead worker's
    backlog cannot turn the nightly digest into an hours-long batch job. The elapsed time
    is re-checked between batches (not just once up front), so the budget is actually
    respected rather than merely advisory.
    """
    from .asr import get_backend
    from .pipeline import process_pending

    backend = get_backend(cfg)
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        if not process_pending(conn, cfg, backend, limit=_CATCHUP_BATCH):
            break


def _catch_up_and_count_missing(conn: sqlite3.Connection, cfg: Config, day: str) -> int:
    """Run the catch-up pass (if enabled and there is anything to catch up on), then
    return how many of `day`'s recordings are still pending or errored."""
    from .timeutil import local_day_bounds

    start_utc, end_utc = local_day_bounds(day, cfg.summarize.timezone)
    pending, errored = _pending_and_errored_counts(conn, start_utc, end_utc)

    budget_s = cfg.schedule.digest_catchup_budget_s
    if pending and budget_s > 0:
        _catch_up_pending(conn, cfg, budget_s)
        pending, errored = _pending_and_errored_counts(conn, start_utc, end_utc)

    return pending + errored


def _insert_incomplete_banner(markdown: str, missing: int) -> str:
    """Insert the incomplete-day banner right after the digest's top-level heading."""
    from .deliver.notify import incomplete_banner

    banner = incomplete_banner(missing)
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join([lines[0], "", banner, *lines[1:]])
    return banner + "\n\n" + markdown


def _incomplete_stdout_note(day: str, missing: int) -> str:
    return f"警告: {day} は {missing} 件の録音が未処理/エラーのため、記録は不完全です。"


@app.command()
def digest(
    day: str = typer.Option(  # noqa: B008
        None, "--day", help="YYYY-MM-DD date (default: today in configured timezone)"
    ),
    deliver: bool = typer.Option(  # noqa: B008
        False,
        "--deliver/--no-deliver",
        help="Send to configured channels (default: print to stdout)",
    ),
    force: bool = typer.Option(False, "--force", help="Resend even if already delivered"),  # noqa: B008
    channel: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--channel",
        help="Specific channel(s) to deliver to: slack, email, repo (can repeat)",
    ),
) -> None:
    """Generate or deliver a daily summary digest.

    By default, generates the digest for today (in the configured timezone) and prints
    the markdown to stdout. Pass --deliver to send to enabled channels instead.

    Examples:
        vas digest
        vas digest --deliver
        vas digest --day 2026-09-14
        vas digest --deliver --channel slack --channel email
        vas digest --deliver --channel repo
    """
    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    # Determine the day if not specified
    if day is None:
        from .timeutil import today_local

        day = today_local(cfg.summarize.timezone)

    # Catch up on this day's backlog (if the worker fell behind), under a wall-clock
    # budget - then find out whether anything is still pending or errored.
    missing = _catch_up_and_count_missing(conn, cfg, day)

    # Generate the digest
    from .summarize import run_day

    markdown = run_day(conn, cfg, day, force=force)

    if missing:
        markdown = _insert_incomplete_banner(markdown, missing)
        typer.echo(_incomplete_stdout_note(day, missing))

    if not deliver:
        # Just print to stdout
        typer.echo(markdown)
    else:
        # Deliver to channels
        from .deliver import deliver_digest

        if not channel:
            # No specific channels requested; use enabled ones from config
            if not cfg.deliver.slack and not cfg.deliver.email and not cfg.deliver.repo:
                typer.echo("No delivery channels enabled in config.")
                typer.echo("Set [deliver] slack=true, email=true, or repo=true in config.toml,")
                typer.echo("and provide secrets: VAS_SLACK_WEBHOOK_URL, VAS_SMTP_PASSWORD")
                raise typer.Exit(1)

        results = deliver_digest(conn, cfg, day, markdown, channels=channel or None, force=force)

        for ch, status in results.items():
            typer.echo(f"{ch}: {status}")


@app.command()
def install_launchd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands, don't run them"),
) -> None:
    """Install macOS launchd services for background worker and daily digest.

    On non-macOS, writes plist files and prints the launchctl commands needed.
    """
    from .launchd import install

    cfg = load_config()
    plist_paths = install(cfg, dry_run=dry_run)
    verb = "Would install" if dry_run else "Installed"
    for path in plist_paths:
        typer.echo(f"{verb}: {path}")


@app.command()
def uninstall_launchd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands, don't run them"),
) -> None:
    """Uninstall macOS launchd services for worker and digest."""
    from .launchd import uninstall

    cfg = load_config()
    uninstall(cfg, dry_run=dry_run)
    typer.echo("Launchd services uninstalled.")
