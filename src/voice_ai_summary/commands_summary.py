"""CLI subcommands for the summary stage. Registered on `app` at import time."""

from __future__ import annotations

import typer

from .cli import app


@app.command()
def episodes(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
) -> None:
    """List episodes for a local day: time range, kind, and utterance count."""
    from datetime import UTC, datetime

    from .config import load_config
    from .db import connect
    from .episodes import build_episodes
    from .recorder_state import format_intervals, pause_intervals
    from .timeutil import fmt_hm, local_time_to_utc, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)

    episode_ids = build_episodes(conn, cfg, day)

    day_start_utc = local_time_to_utc(day, "00:00", tz)
    day_end_utc = local_time_to_utc(day, "24:00", tz)
    intervals = pause_intervals(cfg.paths.root, day_start_utc, day_end_utc, now=datetime.now(UTC))

    if not episode_ids:
        typer.echo(f"{day}: no episodes")
        for interval_start, interval_end in intervals:
            typer.echo(
                f"--- recorder paused {format_intervals([(interval_start, interval_end)], tz)} ---"
            )
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

    remaining = list(intervals)
    for row in rows:
        while remaining and remaining[0][1] <= row["started_at_utc"]:
            interval_start, interval_end = remaining.pop(0)
            typer.echo(
                f"--- recorder paused {format_intervals([(interval_start, interval_end)], tz)} ---"
            )
        start = fmt_hm(row["started_at_utc"], tz)
        end = fmt_hm(row["ended_at_utc"], tz)
        title = row["title"] or "(untitled)"
        typer.echo(
            f"#{row['id']} {start}-{end} [{row['kind']}] {title} "
            f"({row['n_utterances']} utterances, {row['source_mix']})"
        )
    for interval_start, interval_end in remaining:
        typer.echo(
            f"--- recorder paused {format_intervals([(interval_start, interval_end)], tz)} ---"
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
def digest_path(
    day: str = typer.Option(None, "--day", help="Local date YYYY-MM-DD (default: today)."),
) -> None:
    """Print the file path of a stored digest, for opening or reading it locally."""
    from .config import load_config
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    day = day or today_local(cfg.summarize.timezone)
    path = cfg.paths.digests / f"{day}.md"
    if not path.is_file():
        typer.echo(f"No digest file for {day}. Run `vas digest --day {day}` first.", err=True)
        raise typer.Exit(code=1)
    typer.echo(str(path))


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
