"""CLI subcommands. Kept separate from cli.py so heavy imports stay lazy inside each command."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli import app
from .config import load_config
from .db import connect


@app.command()
def status() -> None:
    """Show data directory, counts, and pending work."""
    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    rec = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]
    utt = conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"]
    inbox = 0
    if cfg.paths.inbox.exists():
        inbox = sum(1 for p in cfg.paths.inbox.iterdir() if p.is_file() and p.suffix != ".json")
    typer.echo(f"data_dir : {cfg.paths.root}")
    typer.echo(f"db       : {cfg.paths.db_path}")
    typer.echo(f"asr backend: {cfg.asr.backend}")
    typer.echo(f"asr model: {cfg.asr.resolved_model}")
    typer.echo(f"inbox files      : {inbox}")
    # Silent transcription failures look exactly like a quiet day in the counts above, so
    # surface the shape that means "we heard speech and wrote nothing down": a settings
    # change once emptied every recording for 17 hours before anyone noticed.
    silent = conn.execute(
        """
        SELECT COUNT(*) AS n FROM recordings r
        WHERE r.processed_at IS NOT NULL
          AND EXISTS (SELECT 1 FROM segments s WHERE s.recording_id = r.id)
          AND NOT EXISTS (SELECT 1 FROM utterances u WHERE u.recording_id = r.id)
        """
    ).fetchone()["n"]
    typer.echo(f"recordings       : {rec} (pending: {pending})")
    typer.echo(f"utterances       : {utt}")
    typer.echo(
        f"speech, no text  : {silent}"
        + ("  <- ASR は無音でないのに何も返していません" if silent else "")
    )


@app.command()
def config_path() -> None:
    """Print the config file path that would be loaded."""
    import os

    from .config import DEFAULT_CONFIG_PATH

    typer.echo(str(Path(os.environ.get("VAS_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()))


@app.command()
def usage(
    days: int = typer.Option(30, "--days", help="Look back this many days."),
) -> None:
    """Show Claude API token usage and estimated cost recorded by this tool."""
    from .llm import USAGE_FILENAME, summarize_usage

    cfg = load_config()
    rows = summarize_usage(cfg.paths.root / USAGE_FILENAME, days=days)
    if not rows:
        typer.echo("no API usage recorded")
        return
    typer.echo(f"{'purpose':<10} {'model':<20} {'calls':>5} {'in':>9} {'out':>8} {'USD':>8}")
    total = 0.0
    for r in rows:
        total += r["usd"]
        tokens_in = r["input"] + r["cache_read"] + r["cache_write"]
        typer.echo(
            f"{r['purpose']:<10} {r['model']:<20} {r['calls']:>5} "
            f"{tokens_in:>9} {r['output']:>8} {r['usd']:>8.3f}"
        )
    typer.echo(f"{'total':<10} {'':<20} {'':>5} {'':>9} {'':>8} {total:>8.3f}")
    typer.echo("(estimate from list prices; only calls made by vas are counted)")
