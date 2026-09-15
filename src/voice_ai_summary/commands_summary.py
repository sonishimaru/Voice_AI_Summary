"""CLI subcommands for the summary stage. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app


@app.command()
def episodes(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
) -> None:
    """List episodes for a local day: time range, kind, and utterance count."""
    from .config import load_config
    from .db import connect
    from .episodes import build_episodes
    from .timeutil import fmt_hm, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)

    episode_ids = build_episodes(conn, cfg, day)
    if not episode_ids:
        typer.echo(f"{day}: no episodes")
        return

    placeholders = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"""
        SELECT e.id, e.started_at_utc, e.ended_at_utc, e.kind, e.title, e.source_mix,
               COUNT(u.id) AS n_utterances
        FROM episodes e
        LEFT JOIN utterances u ON u.episode_id = e.id
        WHERE e.id IN ({placeholders})
        GROUP BY e.id
        ORDER BY e.started_at_utc
        """,
        episode_ids,
    ).fetchall()
    for row in rows:
        start = fmt_hm(row["started_at_utc"], tz)
        end = fmt_hm(row["ended_at_utc"], tz)
        title = row["title"] or "(untitled)"
        typer.echo(
            f"#{row['id']} {start}-{end} [{row['kind']}] {title} "
            f"({row['n_utterances']} utterances, {row['source_mix']})"
        )


@app.command()
def summarize(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
    force: bool = typer.Option(False, "--force", help="Recompute even if summaries exist."),
) -> None:
    """Summarize a local day: build episodes, extract, roll up, and write the digest."""
    from .config import load_config
    from .db import connect
    from .summarize import run_day
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    run_day(conn, cfg, day, force=force)
    typer.echo(str(cfg.paths.digests / f"{day}.md"))


@app.command()
def correct(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
    force: bool = typer.Option(False, "--force", help="Re-correct even if already corrected."),
    show: bool = typer.Option(
        False, "--show", help="Print a before/after diff of corrected lines."
    ),
) -> None:
    """Run the Claude correction pass over a local day's ASR text."""
    from .config import load_config
    from .correct import correct_day
    from .db import connect
    from .timeutil import local_day_bounds, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    changed = correct_day(conn, cfg, day, force=force)
    typer.echo(f"{day}: corrected {changed} utterance(s)")

    if show:
        start_utc, end_utc = local_day_bounds(day, cfg.summarize.timezone)
        rows = conn.execute(
            "SELECT raw_text, text FROM utterances"
            " WHERE raw_text IS NOT NULL AND abs_start_utc >= ? AND abs_start_utc < ?"
            " ORDER BY abs_start_utc",
            (start_utc, end_utc),
        ).fetchall()
        for row in rows:
            typer.echo(f"{row['raw_text']} → {row['text']}")


@app.command()
def show(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
) -> None:
    """Print the stored Markdown digest for a local day."""
    from .config import load_config
    from .db import connect
    from .summarize import PROMPT_VERSION
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    row = conn.execute(
        "SELECT markdown FROM summaries WHERE scope='day' AND scope_key=? AND prompt_version=?",
        (day, PROMPT_VERSION),
    ).fetchone()
    if row is None or row["markdown"] is None:
        typer.echo(f"No summary stored for {day}. Run `vas summarize --day {day}` first.")
        raise typer.Exit(code=1)
    typer.echo(row["markdown"])
