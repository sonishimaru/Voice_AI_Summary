"""Anthropic client construction and per-call usage accounting.

The key is read from `VAS_ANTHROPIC_API_KEY` first so this tool never shares a key with
other software on the machine (Claude Code bills to `ANTHROPIC_API_KEY` when it is set).
Every call records `response.usage` to `<data_dir>/usage.jsonl`; `vas usage` prices it.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic

from .config import Config

USAGE_FILENAME = "usage.jsonl"

# USD per million tokens: (input, output). Cache reads cost 0.1x input, cache writes 1.25x.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
}

_usage_path: Path | None = None
_daily_budget_usd: float | None = None


class BudgetExceeded(RuntimeError):
    """Today's recorded API spend is over `[llm] daily_budget_usd`."""


def friendly_api_error(exc: anthropic.APIStatusError, model: str) -> RuntimeError | None:
    """Turn a setup problem (billing, auth, unknown model) into an actionable message.

    Returns None for anything that is genuinely unexpected, which keeps real bugs loud.
    """
    message = str(getattr(exc, "message", "") or exc)
    if "credit balance" in message.lower():
        return RuntimeError(
            "Anthropic API credit balance is empty. Add credits at "
            "https://platform.claude.com/settings/billing (this is separate from a "
            "Claude.ai subscription), then run the command again."
        )
    if exc.status_code in (401, 403):
        return RuntimeError(
            "The Anthropic API rejected the key in VAS_ANTHROPIC_API_KEY "
            f"(HTTP {exc.status_code}). Check that it is a valid API key."
        )
    if exc.status_code == 404:
        return RuntimeError(f"Unknown model {model!r}. Check the model names in config.toml.")
    return None


def api_key() -> str | None:
    return os.environ.get("VAS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def make_client(cfg: Config) -> anthropic.Anthropic:
    """Build the client and point usage tracking at this data directory."""
    global _usage_path, _daily_budget_usd
    _usage_path = cfg.paths.root / USAGE_FILENAME
    _daily_budget_usd = cfg.llm.daily_budget_usd
    _enforce_budget()
    key = api_key()
    if not key:
        raise RuntimeError(
            "No API key: set VAS_ANTHROPIC_API_KEY (preferred) or ANTHROPIC_API_KEY."
        )
    return anthropic.Anthropic(api_key=key)


def set_usage_path(path: Path | None, *, daily_budget_usd: float | None = None) -> None:
    global _usage_path, _daily_budget_usd
    _usage_path = path
    _daily_budget_usd = daily_budget_usd


def spent_today(path: Path) -> float:
    """Estimated USD recorded since local midnight."""
    if not path.is_file():
        return 0.0
    today = datetime.now().astimezone().date()
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        at = datetime.strptime(rec["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if at.astimezone().date() == today:
            total += price_record(rec)
    return total


def check_budget() -> None:
    """Raise `BudgetExceeded` when today's recorded spend is already over the cap.

    Called before each request, so a response that has been paid for is always
    returned and the run stops on the *next* call instead of discarding work.
    """
    _enforce_budget()


def _enforce_budget() -> None:
    if _usage_path is None or _daily_budget_usd is None:
        return
    spent = spent_today(_usage_path)
    if spent > _daily_budget_usd:
        raise BudgetExceeded(
            f"today's Claude API spend (${spent:.2f}) exceeds [llm] daily_budget_usd "
            f"(${_daily_budget_usd:.2f}); raise the limit in config.toml to continue"
        )


def track_usage(purpose: str, model: str, response: Any) -> None:
    """Append one usage record. Silently a no-op when tracking is not configured."""
    usage = getattr(response, "usage", None)
    if _usage_path is None or usage is None:
        return
    record = {
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": purpose,
        "model": model,
        "input": getattr(usage, "input_tokens", 0) or 0,
        "output": getattr(usage, "output_tokens", 0) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    _usage_path.parent.mkdir(parents=True, exist_ok=True)
    with _usage_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Used for models missing from PRICES: over-estimating keeps `daily_budget_usd` a real
# ceiling instead of silently disabling it on an unrecognised model name.
_FALLBACK_PRICE = max(PRICES.values())


def price_record(rec: dict) -> float:
    """Estimated USD for one record; unknown models are priced at the top of the table."""
    inp, out = PRICES.get(rec["model"], _FALLBACK_PRICE)
    return (
        rec["input"] * inp
        + rec["cache_write"] * inp * 1.25
        + rec["cache_read"] * inp * 0.1
        + rec["output"] * out
    ) / 1_000_000


def summarize_usage(path: Path, *, days: int = 30) -> list[dict]:
    """Aggregate usage.jsonl by (purpose, model) for the last `days` days."""
    if not path.is_file():
        return []
    since = datetime.now(UTC) - timedelta(days=days)
    totals: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "usd": 0.0}
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if datetime.strptime(rec["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) < since:
            continue
        t = totals[(rec["purpose"], rec["model"])]
        t["calls"] += 1
        for k in ("input", "output", "cache_write", "cache_read"):
            t[k] += rec.get(k, 0)
        t["usd"] += price_record(rec)
    return [
        {"purpose": p, "model": m, **v}
        for (p, m), v in sorted(totals.items(), key=lambda kv: -kv[1]["usd"])
    ]
