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
    typer.echo(f"asr model: {cfg.asr.model}")
    typer.echo(f"inbox files      : {inbox}")
    typer.echo(f"recordings       : {rec} (pending: {pending})")
    typer.echo(f"utterances       : {utt}")


@app.command()
def config_path() -> None:
    """Print the config file path that would be loaded."""
    import os

    from .config import DEFAULT_CONFIG_PATH

    typer.echo(str(Path(os.environ.get("VAS_CONFIG", str(DEFAULT_CONFIG_PATH))).expanduser()))
