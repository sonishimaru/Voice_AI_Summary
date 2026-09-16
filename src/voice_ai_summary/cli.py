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
def main() -> None:
    """Console entry point: report the tool's own errors as messages, not tracebacks."""
    from .llm import BudgetExceeded

    try:
        app()
    except BudgetExceeded as exc:
        typer.secho(f"stopped: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from None
    except RuntimeError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise SystemExit(1) from None


from . import (  # noqa: E402
    commands,  # noqa: F401
    commands_deliver,  # noqa: F401
    commands_pipeline,  # noqa: F401
    commands_summary,  # noqa: F401
    commands_vocab,  # noqa: F401
)
