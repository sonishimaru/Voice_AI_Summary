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
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass
class HardenChange:
    """One file or directory whose mode `harden` changed (or would, under dry-run)."""

    path: Path
    old_mode: int
    new_mode: int


def _harden_one(path: Path, desired_mode: int, *, dry_run: bool) -> HardenChange | None:
    """Chmod `path` to `desired_mode` if it differs, reporting the change.

    Symlinks are skipped entirely (never followed, never chmod'd). A missing path is
    silently skipped. A failed chmod is logged to stderr and swallowed -- `harden` must
    get through the rest of the tree even if one entry (e.g. owned by another user)
    can't be fixed.
    """
    try:
        if path.is_symlink():
            return None
        old_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None
    if old_mode == desired_mode:
        return None
    if not dry_run:
        try:
            path.chmod(desired_mode)
        except OSError as exc:
            print(f"harden: failed to chmod {path}: {exc}", file=sys.stderr)
            return None
    return HardenChange(path=path, old_mode=old_mode, new_mode=desired_mode)


def _harden_tree(root: Path, *, dry_run: bool) -> list[HardenChange]:
    """Recursively harden `root`: directories to 0700, files to 0600.

    Symlinked directories are not descended into (`followlinks=False`) and are, like
    any other symlink, skipped by `_harden_one` rather than chmod'd.
    """
    changes: list[HardenChange] = []
    if (change := _harden_one(root, 0o700, dry_run=dry_run)) is not None:
        changes.append(change)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in dirnames:
            if (change := _harden_one(base / name, 0o700, dry_run=dry_run)) is not None:
                changes.append(change)
        for name in filenames:
            if (change := _harden_one(base / name, 0o600, dry_run=dry_run)) is not None:
                changes.append(change)
    return changes


def harden(
    cfg,
    *,
    dry_run: bool = False,
    log_dir: Path | None = None,
    key_path: Path | None = None,
) -> list[HardenChange]:
    """Fix up permissions across everything this tool has ever written to disk.

    Covers `cfg.paths.root` (the whole data tree: db, audio, digests, usage/audit
    logs -- directories to 0700, files to 0600), the digest mirror directory and its
    `*.md` files when `cfg.paths.digest_mirror` is set (the mirror's own parents are
    left alone -- it lives wherever the user pointed it), the Anthropic API key file
    and its parent directory, and the launchd log directory and the log files in it
    (both created at the process's default umask before this module existed). Returns
    only the entries whose mode actually changed (or, under `dry_run`, would have).
    """
    changes: list[HardenChange] = []

    root = cfg.paths.root
    if root.exists():
        changes.extend(_harden_tree(root, dry_run=dry_run))

    mirror = cfg.paths.digest_mirror
    if mirror is not None and mirror.exists():
        if (change := _harden_one(mirror, 0o700, dry_run=dry_run)) is not None:
            changes.append(change)
        for md_file in sorted(mirror.glob("*.md")):
            if (change := _harden_one(md_file, 0o600, dry_run=dry_run)) is not None:
                changes.append(change)

    if key_path is None:
        from .llm import api_key_path

        key_path = api_key_path()
    if key_path.parent.exists():
        if (change := _harden_one(key_path.parent, 0o700, dry_run=dry_run)) is not None:
            changes.append(change)
    if key_path.exists():
        if (change := _harden_one(key_path, 0o600, dry_run=dry_run)) is not None:
            changes.append(change)

    if log_dir is None:
        from .launchd import log_dir as _default_log_dir

        log_dir = _default_log_dir()
    if log_dir.exists():
        if (change := _harden_one(log_dir, 0o700, dry_run=dry_run)) is not None:
            changes.append(change)
        for entry in sorted(log_dir.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                continue
            if (change := _harden_one(entry, 0o600, dry_run=dry_run)) is not None:
                changes.append(change)

    return changes


FILEVAULT_WARNING = (
    "FileVault が無効です。音声と文字起こしはファイル権限だけで守られています。"
    "システム設定 > プライバシーとセキュリティで有効化してください。"
)


def filevault_status() -> str | None:
    """ "on"/"off" from `fdesetup status` on macOS, else `None` (including any failure).

    Not macOS, `fdesetup` missing, a timeout, or output that doesn't match either
    expected phrase all fall through to `None` -- this is advisory, never load-bearing.
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(["fdesetup", "status"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    if "FileVault is On" in output:
        return "on"
    if "FileVault is Off" in output:
        return "off"
    return None


SPOTLIGHT_WARNING = (
    "Spotlight がこのデータを索引しています（{count} 件）。ダイジェストは平文なので"
    "本文まで索引され、他人の発話や取引先名が無関係な検索の結果に出ます。"
    "システム設定 > Spotlight > プライバシー に次を追加してください: {paths}"
)


def spotlight_indexed(path: Path) -> int | None:
    """How many files under `path` Spotlight has indexed, or `None` if unknown.

    File permissions keep this data from other accounts, but they do nothing about
    Spotlight: it reads the digests as the owner and copies their *text* into the
    system index, where it surfaces in results for unrelated searches. That is a
    different exposure from the one `harden` addresses, so it is worth reporting
    even though nothing here can fix it -- excluding a folder needs the Spotlight
    Privacy list (`.metadata_never_index` stopped working in recent macOS).

    Not macOS, no `mdfind`, a timeout, or unparsable output all give `None`; this is
    advisory, never load-bearing.
    """
    if sys.platform != "darwin" or not path.exists():
        return None
    try:
        result = subprocess.run(
            ["mdfind", "-onlyin", str(path), "-count", "kMDItemFSName == '*'"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int((result.stdout or "").strip())
    except ValueError:
        return None


def spotlight_report(cfg) -> str | None:
    """One line about Spotlight indexing of the data directory and digest mirror,
    or `None` when nothing is indexed (or the answer cannot be determined)."""
    indexed: list[Path] = []
    total = 0
    for path in (cfg.paths.root, cfg.paths.digest_mirror):
        if path is None:
            continue
        count = spotlight_indexed(path)
        if count:
            indexed.append(path)
            total += count
    if not indexed:
        return None
    return SPOTLIGHT_WARNING.format(count=total, paths=", ".join(str(p) for p in indexed))


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
