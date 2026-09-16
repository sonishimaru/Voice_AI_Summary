"""CLI subcommands for Claude Desktop integration. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app


@app.command()
def install_desktop(
    bin: str = typer.Option(
        "", "--bin", help="Path to the vas-mcp executable (default: auto-detect)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print what would happen, write nothing."
    ),
) -> None:
    """Register vas as an MCP server in Claude Desktop -- no more terminal needed.

    After this, you talk to Claude Desktop instead of running `vas` commands by hand:
    it can show today's summary, search, reprocess a day, manage the glossary and
    report API usage, all through the MCP server this installs. Quit and reopen Claude
    Desktop afterwards for it to pick up the new server.
    """
    from .desktop import default_bin, ensure_api_key_file, install

    key_status = ensure_api_key_file()
    bin_path = bin or default_bin()
    config_path = install(bin_path, dry_run=dry_run)

    verb = "Would write" if dry_run else "Wrote"
    typer.echo(f"{verb} MCP server entry ({bin_path}) to {config_path}")
    typer.echo(key_status)
    typer.echo("Claude Desktop を一度終了して開き直してください。")


@app.command()
def uninstall_desktop() -> None:
    """Remove vas's MCP server entry from Claude Desktop's config."""
    from .desktop import uninstall

    removed = uninstall()
    if removed:
        typer.echo("Removed voice-ai-summary from Claude Desktop's config.")
    else:
        typer.echo("voice-ai-summary was not registered; nothing to remove.")
