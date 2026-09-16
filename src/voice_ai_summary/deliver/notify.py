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


def headline(markdown: str, *, limit: int = 180) -> str:
    """The first highlight bullet, or the first non-heading line, trimmed for a notification."""
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
        return text[: limit - 1] + "…" if len(text) > limit else text
    return "本文を確認してください"


def _escape(text: str) -> str:
    """Escape for an AppleScript string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def send_notification(
    day: str, markdown: str, *, path: str | None = None, runner: _Runner | None = None
) -> None:
    """Post a macOS notification summarising `day`'s digest.

    Raises `RuntimeError` when `osascript` is unavailable (i.e. not macOS) or fails.
    """
    run = runner if runner is not None else subprocess.run
    subtitle = path or "vas show で全文を表示"
    script = (
        f'display notification "{_escape(headline(markdown))}"'
        f' with title "{_escape(day)} のサマリ"'
        f' subtitle "{_escape(subtitle)}"'
    )
    try:
        result = run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:  # not macOS
        raise RuntimeError("osascript not found; notifications need macOS.") from exc
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError(f"osascript failed: {getattr(result, 'stderr', '')}".strip())
