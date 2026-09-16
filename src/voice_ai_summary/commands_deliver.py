"""CLI subcommands for the deliver stage. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app
from .config import load_config
from .db import connect


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

    # Generate the digest
    from .summarize import run_day

    markdown = run_day(conn, cfg, day, force=force)

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
