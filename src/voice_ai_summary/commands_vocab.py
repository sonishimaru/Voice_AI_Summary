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
    scope: str = typer.Option(
        "mine",
        "--scope",
        help="mine: your own messages | channels: every channel you belong to | all: both.",
    ),
    days: int = typer.Option(90, "--days", help="How many days of Slack history to read."),
    limit: int = typer.Option(
        1500, "--limit", help="Maximum number of your own messages (scope mine/all)."
    ),
    channel_limit: int = typer.Option(
        5000, "--channel-limit", help="Maximum total channel messages (scope channels/all)."
    ),
    per_channel: int = typer.Option(
        300, "--per-channel", help="Maximum messages taken from each channel."
    ),
    channel: list[str] = typer.Option(  # noqa: B008
        [], "--channel", help="Only scan these channel names (repeatable, without #)."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Override the extraction model (default: [correct].model)."
    ),
) -> None:
    """Grow the glossary from Slack: your own messages and/or your channels' conversations.

    Needs `VAS_SLACK_USER_TOKEN`, a Slack *user* token (`xoxp-…`). Scope `mine` uses
    `search:read`; `channels` additionally needs `channels:read`, `groups:read`,
    `channels:history` and `groups:history`.
    """
    import httpx

    from .config import load_config
    from .glossary import extract_glossary, load_glossary, save_glossary
    from .llm import make_client
    from .slack_import import import_channel_messages, import_from_slack

    if scope not in ("mine", "channels", "all"):
        raise typer.BadParameter("--scope must be mine, channels or all")
    cfg = load_config()
    token = cfg.slack_user_token
    if not token:
        typer.echo("VAS_SLACK_USER_TOKEN is not set (needs a Slack user token).")
        raise typer.Exit(code=1)

    own: list[str] = []
    from_channels: list[str] = []
    with httpx.Client(timeout=30.0) as http_client:
        if scope in ("mine", "all"):
            own = import_from_slack(http_client, token, days=days, limit=limit)
            typer.echo(f"fetched {len(own)} of your own message(s)")
        if scope in ("channels", "all"):
            from_channels = import_channel_messages(
                http_client,
                token,
                days=days,
                per_channel=per_channel,
                limit=channel_limit,
                channels=channel or None,
            )
            names = [line[1:] for line in from_channels if line.startswith("#")]
            typer.echo(
                f"fetched {len(from_channels) - len(names)} channel message(s) "
                f"from {len(names)} channel(s): {', '.join(names) or '-'}"
            )
    if not own and not from_channels:
        return

    client = make_client(cfg)
    extraction_model = model or cfg.correct.model
    updated = load_glossary(cfg)
    if own:
        updated = extract_glossary(client, extraction_model, own, updated)
    if from_channels:
        updated = extract_glossary(
            client, extraction_model, from_channels, updated, own_messages=False
        )
    save_glossary(cfg, updated)
    typer.echo(
        f"glossary now has {len(updated.terms)} term(s), {len(updated.style_notes)} style note(s)"
    )
