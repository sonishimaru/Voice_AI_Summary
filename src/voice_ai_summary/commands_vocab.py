"""CLI subcommands for the personal glossary. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app

vocab_app = typer.Typer(
    name="vocab", help="Manage the personal glossary used by ASR, correction and summaries."
)
app.add_typer(vocab_app, name="vocab")


@vocab_app.command("show")
def vocab_show() -> None:
    """Print the current glossary."""
    from .config import load_config
    from .glossary import load_glossary

    cfg = load_config()
    block = load_glossary(cfg).prompt_block()
    typer.echo(block if block else "(glossary is empty)")


@vocab_app.command("add")
def vocab_add(
    term: str = typer.Argument(..., help="The term/name in its correct spelling."),
    alias: list[str] = typer.Option(  # noqa: B008
        [], "--alias", help="Alternate spelling/reading (repeatable)."
    ),
    note: str = typer.Option("", "--note", help="Short note on what/who this is."),
) -> None:
    """Add a term to the glossary, or merge into it if the term already exists."""
    from .config import load_config
    from .glossary import Glossary, Term, load_glossary, save_glossary

    cfg = load_config()
    glossary = load_glossary(cfg)
    updated = glossary.merge(Glossary(terms=[Term(term=term, aliases=list(alias), note=note)]))
    save_glossary(cfg, updated)
    typer.echo(f"added/updated: {term}")


@vocab_app.command("import-slack")
def vocab_import_slack(
    days: int = typer.Option(90, "--days", help="How many days of Slack history to search."),
    limit: int = typer.Option(1500, "--limit", help="Maximum number of messages to fetch."),
    model: str | None = typer.Option(
        None, "--model", help="Override the extraction model (default: [correct].model)."
    ),
) -> None:
    """Build the glossary from the user's own Slack messages.

    Needs `VAS_SLACK_USER_TOKEN`, a Slack *user* token (`xoxp-…`) with the `search:read`
    scope - `search.messages` is not available to bot tokens.
    """
    import httpx

    from .config import load_config
    from .glossary import extract_glossary, load_glossary, save_glossary
    from .llm import make_client
    from .slack_import import import_from_slack

    cfg = load_config()
    token = cfg.slack_user_token
    if not token:
        typer.echo("VAS_SLACK_USER_TOKEN is not set (needs a Slack user token, scope search:read).")
        raise typer.Exit(code=1)

    with httpx.Client(timeout=30.0) as http_client:
        messages = import_from_slack(http_client, token, days=days, limit=limit)
    typer.echo(f"fetched {len(messages)} message(s) from Slack")
    if not messages:
        return

    existing = load_glossary(cfg)
    updated = extract_glossary(make_client(cfg), model or cfg.correct.model, messages, existing)
    save_glossary(cfg, updated)
    typer.echo(
        f"glossary now has {len(updated.terms)} term(s), {len(updated.style_notes)} style note(s)"
    )
