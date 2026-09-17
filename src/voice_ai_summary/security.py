"""File-permission and secret-redaction primitives used across the codebase.

This tool records the user's voice (and other people's) all day, stores verbatim
transcripts and audio, and sends transcripts to the Claude API. A security review found
exactly one explicit file mode in the whole tree (`desktop.ensure_api_key_file`'s
`os.open(..., 0o600)` for the API key) -- everything else (the SQLite database, audio,
digests, glossary, logs) was created at the process's default umask, which is
world-readable on a standard Mac. This module generalises that one-off pattern into the
foundation the rest of the security work builds on: a process-wide umask, private
file/dir helpers, and a secret-redaction helper for anything written to logs or an audit
trail.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

AUDIT_FILENAME = "audit.jsonl"

_MODE_BITS = {
    "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
}

_GROUP_OTHER_MASK = stat.S_IRWXG | stat.S_IRWXO

# Applied in this order: more specific patterns (a full credentialed URL, a Slack
# webhook) must run before the generic `key=value` pattern would otherwise mangle them.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-…[redacted]"),
    (re.compile(r"xox[abpr]-[A-Za-z0-9\-]+"), "xox*-[redacted]"),
    (re.compile(r"(?<=://)[^/\s:@]+:[^@\s]+@"), "***:***@"),
    (
        re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]+"),
        "hooks.slack.com/services/[redacted]",
    ),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[=:]\s*\S+"),
        r"\1=[redacted]",
    ),
]


def apply_process_umask() -> int:
    """Set the process umask to 0o077 and return the previous mask.

    Every file this process creates -- the SQLite database, audio, digests, glossary,
    logs, usage records -- must be private by default rather than relying on each
    call site to remember a mode. Callers are the CLI (`cli.main`), the MCP server and
    the worker/launchd entry points, i.e. every place this process starts running.
    """
    return os.umask(0o077)


def open_private(path: Path, mode: str = "w"):
    """Open `path` for writing with the file created at mode 0600.

    Generalises the `desktop.ensure_api_key_file` pattern (`os.open` with an explicit
    mode, then wrap the fd) to any write or append. Only `"w"` (truncate) and `"a"`
    (append) are accepted; anything else raises `ValueError`. The 0600 mode is applied
    only at creation time by the OS -- an *existing* file keeps whatever mode it already
    has, so if it might have been created world-readable before this helper existed,
    fix it first with `ensure_private_file`.
    """
    if mode not in _MODE_BITS:
        raise ValueError(f"open_private only supports 'w' or 'a', got {mode!r}")
    fd = os.open(str(path), _MODE_BITS[mode], 0o600)
    return os.fdopen(fd, mode, encoding="utf-8")


def ensure_private_dir(path: Path) -> bool:
    """Create `path` (and parents) if needed, and ensure it is not group/other-accessible.

    Returns whether a chmod was actually needed (i.e. the directory was not already
    private) -- a newly created directory inherits the umask, which is 0700 once
    `apply_process_umask` has run, but existing directories created before this module
    existed may still be permissive.
    """
    path.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & _GROUP_OTHER_MASK:
        path.chmod(0o700)
        return True
    return False


def ensure_private_file(path: Path) -> bool:
    """Chmod an existing file to 0600 if it has any group/other bits set.

    Returns whether a chmod happened. A missing file is not an error -- it returns
    False, since there is nothing to fix.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return False
    if mode & _GROUP_OTHER_MASK:
        path.chmod(0o600)
        return True
    return False


def redact(text: str) -> str:
    """Mask secrets in free text before it is stored or shown.

    Applies, in order: Anthropic API keys, Slack tokens, credentials embedded in a URL
    (`user:pass@host`), Slack incoming-webhook URLs, and a generic
    `key=value`/`key: value` pattern for `api_key`/`token`/`password`/`secret`. A plain
    URL with no embedded credentials, and prose that merely mentions "token" (in any
    language) without a `=`/`:` value following it, are left untouched.
    """
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def audit(root: Path, tool: str, args: dict, outcome: str) -> None:
    """Append one redacted JSON audit line to `root / AUDIT_FILENAME`.

    Never raises: any failure (an unwritable `root`, a non-serialisable arg, ...) is
    reported to stderr and swallowed, since a broken audit trail must never break the
    tool call it is trying to record.
    """
    try:
        safe_args = {k: (redact(v) if isinstance(v, str) else v) for k, v in args.items()}
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": tool,
            "args": safe_args,
            "outcome": redact(outcome[:300]),
        }
        with open_private(root / AUDIT_FILENAME, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - audit logging must never raise
        print(f"audit: failed to record {tool!r} outcome: {exc}", file=sys.stderr)
