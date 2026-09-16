"""MCP stdio server exposing `vas` functionality as tools for Claude Desktop.

This lets the person driving `vas` do so entirely from Claude Desktop instead of a
terminal: read digests, search/browse transcripts, rebuild a day's summary, manage the
personal glossary, check pipeline status and API spend, retry failed recordings, and
pull+reinstall the latest code.

IMPORTANT: stdout carries the MCP JSON-RPC protocol itself. Nothing in this module (or
anything it imports) may `print` or otherwise write to stdout - that would corrupt the
stream and break the session. Diagnostics belong on stderr via `logging`, which is safe.

Heavy imports (config, db, summarize, ...) are kept lazy inside each tool function, the
same style used by `commands_*.py`, so importing this module (which `mcp_server.main`
does at process start) stays cheap.
"""

from __future__ import annotations

import functools
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="voice-ai-summary",
    instructions=(
        "Tools for the `vas` voice-ai-summary pipeline: an always-on voice recorder that "
        "locally transcribes Japanese speech and uses Claude to produce daily digests. "
        "Use `daily_summary`/`list_days` to read what has already been summarized, "
        "`search_transcript`/`transcript` to look at the raw local transcript, "
        "`rebuild_day` to (re)run summarization for a day (this calls the Claude API and "
        "costs money), `status`/`api_usage` to check pipeline health and spend, "
        "`list_vocabulary`/`add_vocabulary` to manage the personal glossary, "
        "`retry_failed` to recover recordings that failed to transcribe, and `update_app` "
        "to pull and reinstall the latest `vas` code."
    ),
)

F = TypeVar("F", bound=Callable[..., str])


def _tool_safe(func: F) -> F:
    """Wrap a tool body so any exception it raises becomes a returned string instead of
    propagating and killing the stdio session.

    `RuntimeError` (which `llm.BudgetExceeded` subclasses, and which tools raise
    themselves for user-facing problems) is surfaced verbatim. Anything else becomes
    `f"{type(exc).__name__}: {exc}"`, so a real bug is still legible instead of silently
    swallowed.
    """

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> str:
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - this IS the tool-boundary catch-all
            return f"{type(exc).__name__}: {exc}"

    return wrapper  # type: ignore[return-value]


def _format_utterance_row(row: sqlite3.Row, tz: str) -> str:
    """Render one utterance row as `HH:MM [speaker] text` in local time."""
    from .timeutil import fmt_hm

    return f"{fmt_hm(row['abs_start_utc'], tz)} [{row['speaker']}] {row['text']}"


def _digest_exists(conn: sqlite3.Connection, day: str) -> bool:
    """Whether a day digest is stored (in `summaries`) or written to the digest file."""
    from .summarize import PROMPT_VERSION

    row = conn.execute(
        "SELECT markdown FROM summaries WHERE scope='day' AND scope_key=? AND prompt_version=?",
        (day, PROMPT_VERSION),
    ).fetchone()
    return row is not None and row["markdown"] is not None


@_tool_safe
def daily_summary(day: str = "") -> str:
    """Return the stored Japanese Markdown daily digest for a local day.

    `day` is a local date `YYYY-MM-DD` in the configured timezone; empty means today.
    Read-only and free - never calls the Claude API. Prefers the digest stored in the
    database and falls back to the digest file on disk. If neither exists, says so and
    names `rebuild_day` (which does cost money) as the way to build one.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    if _digest_exists(conn, day):
        from .summarize import PROMPT_VERSION

        row = conn.execute(
            "SELECT markdown FROM summaries WHERE scope='day' AND scope_key=? AND prompt_version=?",
            (day, PROMPT_VERSION),
        ).fetchone()
        return row["markdown"]

    digest_path = cfg.paths.digests / f"{day}.md"
    if digest_path.is_file():
        return digest_path.read_text(encoding="utf-8")

    return (
        f"No digest found for {day}. Call rebuild_day(day={day!r}) to build one - note "
        "that calls the Claude API and costs money."
    )


@_tool_safe
def list_days(limit: int = 30) -> str:
    """List the most recent local days that have any recorded utterances.

    For each of the last `limit` days (default 30) with at least one utterance, shows
    the local date, how many utterances were recorded, and whether a digest already
    exists for it. Local dates are derived from each utterance's UTC timestamp using the
    configured timezone, so a recording made late at night lands on the correct local
    day even when that crosses a UTC day boundary. Read-only and free.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import to_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    rows = conn.execute("SELECT abs_start_utc FROM utterances").fetchall()
    if not rows:
        return "No utterances recorded yet."

    counts: dict[str, int] = {}
    for row in rows:
        day = to_local(row["abs_start_utc"], tz).strftime("%Y-%m-%d")
        counts[day] = counts.get(day, 0) + 1

    days = sorted(counts, reverse=True)[:limit]
    lines = []
    for day in days:
        digest = "yes" if _digest_exists(conn, day) else "no"
        lines.append(f"{day}: {counts[day]} utterance(s), digest: {digest}")
    return "\n".join(lines)


@_tool_safe
def search_transcript(query: str, day: str = "", limit: int = 30) -> str:
    """Full-text search the local voice transcript for `query`.

    Returns matching lines as `HH:MM [me|other|unknown] text` in local time, oldest
    first. `day` (local date `YYYY-MM-DD`) restricts the search to one day; empty
    searches everything. `limit` caps the number of results (default 30). Read-only and
    free. Any punctuation in `query` (quotes, hyphens, etc.) is treated as literal text,
    never as search syntax.
    """
    from .config import load_config
    from .db import connect
    from .search import search as run_search

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone

    rows = run_search(conn, query, limit=limit, day=day or None, tz=tz)
    if not rows:
        where = f" on {day}" if day else ""
        return f"No matches for {query!r}{where}."
    return "\n".join(_format_utterance_row(row, tz) for row in rows)


@_tool_safe
def transcript(day: str = "", limit: int = 800) -> str:
    """Return a local day's whole transcript, oldest first.

    `day` is a local date `YYYY-MM-DD`; empty means today. Lines are formatted as
    `HH:MM [me|other|unknown] text`. Truncated to `limit` utterances (default 800) with
    a trailing note when there are more, so a very long day does not blow the response
    up. Read-only and free.
    """
    from .config import load_config
    from .db import connect
    from .timeutil import local_day_bounds, today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    tz = cfg.summarize.timezone
    day = day or today_local(tz)

    start, end = local_day_bounds(day, tz)
    rows = conn.execute(
        "SELECT abs_start_utc, speaker, text FROM utterances"
        " WHERE abs_start_utc >= ? AND abs_start_utc < ?"
        " ORDER BY abs_start_utc, t_start_ms",
        (start, end),
    ).fetchall()
    if not rows:
        return f"No utterances recorded for {day}."

    shown = rows[:limit]
    lines = [_format_utterance_row(row, tz) for row in shown]
    if len(rows) > limit:
        lines.append(f"... truncated: showing {limit} of {len(rows)} utterances")
    return "\n".join(lines)


@_tool_safe
def rebuild_day(day: str = "", force: bool = False, correct: bool = True) -> str:
    """Rebuild a local day's summary and return the resulting Markdown digest.

    THIS CALLS THE CLAUDE API AND COSTS MONEY (subject to `[llm] daily_budget_usd` in
    config.toml; check `api_usage` for recent spend). `day` is a local date
    `YYYY-MM-DD`, default today. Unless `correct=False`, first runs the Claude
    correction pass over the day's ASR text, then (re)builds episodes and the daily
    digest. Summaries already cached for unchanged content are reused unless
    `force=True`, which recomputes everything for the day regardless of caching.
    """
    from .config import load_config
    from .correct import correct_day
    from .db import connect
    from .summarize import run_day
    from .timeutil import today_local

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    day = day or today_local(cfg.summarize.timezone)

    if correct:
        correct_day(conn, cfg, day, force=force)
    return run_day(conn, cfg, day, force=force)


@_tool_safe
def status() -> str:
    """Show `vas` pipeline health: data directory, database path, ASR model, inbox
    file count, recording counts (total, pending, and errored), and utterance count.

    Read-only and free - never touches the network.
    """
    from .config import load_config
    from .db import connect

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    rec = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]
    errored = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE error IS NOT NULL"
    ).fetchone()["n"]
    utt = conn.execute("SELECT COUNT(*) AS n FROM utterances").fetchone()["n"]
    inbox = 0
    if cfg.paths.inbox.exists():
        inbox = sum(1 for p in cfg.paths.inbox.iterdir() if p.is_file() and p.suffix != ".json")

    return "\n".join(
        [
            f"data_dir : {cfg.paths.root}",
            f"db       : {cfg.paths.db_path}",
            f"asr model: {cfg.asr.model}",
            f"inbox files      : {inbox}",
            f"recordings       : {rec} (pending: {pending}, errored: {errored})",
            f"utterances       : {utt}",
        ]
    )


@_tool_safe
def api_usage(days: int = 30) -> str:
    """Show Claude API token usage and estimated cost recorded by `vas`.

    Aggregates over the last `days` days (default 30) by purpose (map/reduce/correct/
    glossary) and model. Estimated from list prices; only calls made by this tool are
    counted. Read-only and free - it reads a local log, it does not call the API.
    """
    from .config import load_config
    from .llm import USAGE_FILENAME, summarize_usage

    cfg = load_config()
    rows = summarize_usage(cfg.paths.root / USAGE_FILENAME, days=days)
    if not rows:
        return "no API usage recorded"

    lines = [f"{'purpose':<10} {'model':<20} {'calls':>5} {'in':>9} {'out':>8} {'USD':>8}"]
    total = 0.0
    for r in rows:
        total += r["usd"]
        tokens_in = r["input"] + r["cache_read"] + r["cache_write"]
        lines.append(
            f"{r['purpose']:<10} {r['model']:<20} {r['calls']:>5} "
            f"{tokens_in:>9} {r['output']:>8} {r['usd']:>8.3f}"
        )
    lines.append(f"{'total':<10} {'':<20} {'':>5} {'':>9} {'':>8} {total:>8.3f}")
    lines.append("(estimate from list prices; only calls made by vas are counted)")
    return "\n".join(lines)


@_tool_safe
def list_vocabulary() -> str:
    """Show the personal glossary: proper nouns/aliases/notes and notation style rules
    used to bias ASR, the correction pass, and summarization. Read-only and free.
    """
    from .config import load_config
    from .glossary import load_glossary

    cfg = load_config()
    block = load_glossary(cfg).prompt_block()
    return block if block else "(glossary is empty)"


@_tool_safe
def add_vocabulary(term: str, aliases: list[str] | None = None, note: str = "") -> str:
    """Add a term to the personal glossary, merging into an existing entry if the term
    (or one of its aliases) is already known.

    `term` is the correct spelling of a name, product, project, or piece of jargon.
    `aliases` (optional) are alternate spellings or readings that should resolve to the
    same entry. `note` (optional) is a short description of what/who it is. Affects
    future ASR, correction, and summarization only - it does not rebuild anything
    already generated. Free - no network call.
    """
    from .config import load_config
    from .glossary import Glossary, Term, load_glossary, save_glossary

    cfg = load_config()
    glossary = load_glossary(cfg)
    new_term = Term(term=term, aliases=list(aliases or []), note=note)
    updated = glossary.merge(Glossary(terms=[new_term]))
    save_glossary(cfg, updated)
    return f"added/updated: {term}"


@_tool_safe
def retry_failed() -> str:
    """Re-queue recordings that previously failed to transcribe, then process them again.

    Clears the error flag (recovering a `.part` file left by an in-progress recorder if
    needed) and re-runs local ASR on them. Local-only - no network call, no cost.
    """
    from .asr import get_backend
    from .config import load_config
    from .db import connect
    from .pipeline import process_pending
    from .pipeline import retry_failed as run_retry_failed

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)

    ids = run_retry_failed(conn, cfg)
    if not ids:
        return "no failed recordings"

    backend = get_backend(cfg)
    count = process_pending(conn, cfg, backend)
    return f"re-queued {len(ids)} recording(s): {ids}\nprocessed {count} recording(s)"


def _repo_dir() -> Path:
    """The git work tree this package is installed from, derived from `__file__`
    (`<repo>/src/voice_ai_summary/mcp_server.py`) rather than the current directory,
    since an MCP server started by Claude Desktop has no meaningful cwd."""
    return Path(__file__).resolve().parents[2]


def _run(cmd: tuple[str, ...], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


@_tool_safe
def update_app() -> str:
    """Pull the latest `vas` code and reinstall it, so updates never need a terminal.

    Runs, in this package's git checkout: `git fetch origin <branch>`, then a
    fast-forward-only merge (refuses and stops cleanly if that is not possible, e.g.
    there are local changes), then `uv pip install -e .` (falling back to
    `python -m pip install -e .` if `uv` is unavailable). The branch is read from the
    `VAS_UPDATE_BRANCH` environment variable, defaulting to `claude/clever-sagan-23tn2w`.
    Refuses if this package is not installed from a git work tree. Restart Claude
    Desktop after a successful update so it picks up the new code.
    """
    import os

    repo_dir = _repo_dir()
    if not (repo_dir / ".git").exists():
        return f"{repo_dir} is not a git work tree; refusing to update."

    branch = os.environ.get("VAS_UPDATE_BRANCH", "claude/clever-sagan-23tn2w")
    output: list[str] = []

    for cmd in (
        ("git", "fetch", "origin", branch),
        ("git", "merge", "--ff-only", f"origin/{branch}"),
    ):
        result = _run(cmd, repo_dir, timeout=60)
        output.append(f"$ {' '.join(cmd)}\n{(result.stdout + result.stderr).strip()}")
        if result.returncode != 0:
            output.append(
                "Update stopped: the step above failed (a fast-forward merge needs no "
                "local changes on this branch). Resolve it in a terminal, then retry."
            )
            return "\n\n".join(output)

    install_cmd: tuple[str, ...] = ("uv", "pip", "install", "-e", ".")
    try:
        result = _run(install_cmd, repo_dir, timeout=300)
    except FileNotFoundError:
        result = None
    if result is None or result.returncode != 0:
        install_cmd = (sys.executable, "-m", "pip", "install", "-e", ".")
        result = _run(install_cmd, repo_dir, timeout=300)

    output.append(f"$ {' '.join(install_cmd)}\n{(result.stdout + result.stderr).strip()}")
    if result.returncode != 0:
        output.append("Install failed; see output above.")
        return "\n\n".join(output)

    output.append("Update complete. Restart Claude Desktop to use the new code.")
    return "\n\n".join(output)


for _fn in (
    daily_summary,
    list_days,
    search_transcript,
    transcript,
    rebuild_day,
    status,
    api_usage,
    list_vocabulary,
    add_vocabulary,
    retry_failed,
    update_app,
):
    server.add_tool(_fn)
del _fn


def main() -> None:
    """Entry point for the `vas-mcp` console script: run the stdio MCP server."""
    server.run("stdio")


if __name__ == "__main__":
    main()
