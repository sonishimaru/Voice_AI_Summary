"""`vas` command line entry point. Subcommands are attached by the pipeline/summary modules."""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    name="vas",
    help="Voice AI Summary: ingest audio, transcribe locally, summarise, deliver daily.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# Subcommand modules register themselves on `app` at import time.
from . import commands  # noqa: E402,F401
