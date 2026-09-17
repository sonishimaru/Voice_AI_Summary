"""CLI subcommands for the personal glossary. Registered on `app` at import time."""

from __future__ import annotations

from pathlib import Path

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
    from .glossary import Glossary, load_glossary, save_glossary, validate_term

    try:
        validated = validate_term(term, list(alias), note)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None

    cfg = load_config()
    glossary = load_glossary(cfg)
    updated = glossary.merge(Glossary(terms=[validated]))
    save_glossary(cfg, updated)
    typer.echo(f"added/updated: {validated.term}")


@vocab_app.command("import")
def vocab_import(
    path: Path = typer.Argument(..., help="JSON file in the glossary.json format."),  # noqa: B008
) -> None:
    """Merge a glossary JSON file (e.g. produced by another tool) into the glossary.

    Accepts `{"terms": [{"term", "aliases", "note"}], "style_notes": [...]}` or a bare list
    of term objects. Existing entries are kept; aliases and notes are merged.

    Every incoming term is validated the same way `vas vocab add` validates its
    argument (`validate_term`); a single invalid entry aborts the whole import with no
    partial write, so a bad file can be fixed and re-run rather than silently truncated.
    Style notes go through the equivalent `validate_style_note`.
    """
    import json

    from .config import load_config
    from .glossary import Glossary, load_glossary, save_glossary, validate_style_note, validate_term

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"terms": data}
    try:
        terms = []
        for t in data.get("terms", []):
            term = t.get("term")
            if not term:
                continue
            aliases = t.get("aliases") or []
            if not isinstance(aliases, list):
                raise ValueError(f"aliases for {term!r} must be a list")
            note = t.get("note") or ""
            terms.append(validate_term(term, aliases, note))
        style_notes = [validate_style_note(str(n)) for n in data.get("style_notes", []) if n]
        incoming = Glossary(terms=terms, style_notes=style_notes)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    cfg = load_config()
    before = load_glossary(cfg)
    updated = before.merge(incoming)
    save_glossary(cfg, updated)
    typer.echo(
        f"imported {len(incoming.terms)} term(s), {len(incoming.style_notes)} style note(s); "
        f"glossary now has {len(updated.terms)} term(s) "
        f"(+{len(updated.terms) - len(before.terms)})"
    )


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
    if not cfg.glossary.slack_import_enabled:
        typer.echo(
            "Slack からの取り込みは無効です。再取り込みが必要なら config.toml の "
            "[glossary] slack_import_enabled = true を設定してください"
            "（参加チャンネル全部のメッセージが Claude API に送られます）。"
        )
        raise typer.Exit(code=1)
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
