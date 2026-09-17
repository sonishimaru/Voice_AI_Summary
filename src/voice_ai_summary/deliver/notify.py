"""macOS notification delivery: tells the user the digest is ready, without sending it anywhere.

This is the local-only channel. The digest itself stays in `<data_dir>/digests/`; only a
short headline reaches the notification centre, and nothing leaves the machine.
"""

from __future__ import annotations

import re
import subprocess
from typing import Protocol


class _Runner(Protocol):
    def __call__(self, args: list[str], **kwargs: object) -> object: ...


# Digest headings are fixed by the reduce prompt; the first bullet under ハイライト is the
# one line worth surfacing.
_HIGHLIGHT_HEADING = "## ハイライト"

# Embedded near the top of an incomplete day's digest Markdown by `commands_deliver`
# (before the `## ハイライト` section, so it never gets confused with a real highlight).
# `headline()` recognises this prefix and surfaces it ahead of the regular highlight, so
# the notification says the day is incomplete too - without `deliver_digest` having to
# pass anything extra through to `send_notification`, whose call signature is unchanged.
_INCOMPLETE_PREFIX = "[未処理あり]"


def incomplete_banner(missing: int) -> str:
    """A line noting `missing` recordings are still pending/errored for the day.

    See `_INCOMPLETE_PREFIX` for how this reaches the notification headline too.
    """
    return f"{_INCOMPLETE_PREFIX} 未処理の録音が{missing}件あるため、この記録は不完全です。"


def _truncate(text: str, limit: int) -> str:
    return text[: limit - 1] + "…" if len(text) > limit else text


def headline(markdown: str, *, limit: int = 180) -> str:
    """The incomplete-day banner if present, else the first highlight bullet, else the
    first non-heading line, trimmed for a notification."""
    for line in markdown.splitlines():
        text = line.strip()
        if text.startswith(_INCOMPLETE_PREFIX):
            return _truncate(text, limit)

    lines = markdown.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == _HIGHLIGHT_HEADING:
            start = i + 1
            break
    for line in lines[start:]:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        text = re.sub(r"^[-*+]\s*", "", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        return _truncate(text, limit)
    return "本文を確認してください"


def _escape(text: str) -> str:
    """Escape for an AppleScript string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def send_desktop_notification(
    message: str, *, title: str, subtitle: str | None = None, runner: _Runner | None = None
) -> None:
    """Post a plain macOS notification. Not digest-shaped: just a title/message/subtitle.

    Used both by `send_notification` below (digest-ready) and by the worker's backlog
    alert. Raises `RuntimeError` when `osascript` is unavailable (i.e. not macOS) or fails.
    """
    run = runner if runner is not None else subprocess.run
    script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
    if subtitle:
        script += f' subtitle "{_escape(subtitle)}"'
    try:
        result = run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:  # not macOS
        raise RuntimeError("osascript not found; notifications need macOS.") from exc
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError(f"osascript failed: {getattr(result, 'stderr', '')}".strip())


def send_notification(
    day: str, markdown: str, *, path: str | None = None, runner: _Runner | None = None
) -> None:
    """Post a macOS notification summarising `day`'s digest.

    Raises `RuntimeError` when `osascript` is unavailable (i.e. not macOS) or fails.
    """
    subtitle = path or "vas show で全文を表示"
    send_desktop_notification(
        headline(markdown), title=f"{day} のサマリ", subtitle=subtitle, runner=runner
    )
