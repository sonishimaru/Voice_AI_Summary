"""Register the `vas-mcp` MCP server with the Claude Desktop app.

Claude Desktop discovers MCP servers from a JSON config file it reads on startup. This
module merges our single entry into that file without disturbing any other server the
user has configured there, and manages a key file the MCP server reads its Anthropic
API key from (Claude Desktop launches the server with a bare environment, so nothing in
the user's shell profile is visible to it -- see `llm.api_key_path()`).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

CONFIG_PATH = Path("~/Library/Application Support/Claude/claude_desktop_config.json")
SERVER_KEY = "voice-ai-summary"


def _config_path(path: Path | None = None) -> Path:
    return path if path is not None else CONFIG_PATH.expanduser()


def _get_env_overrides() -> dict[str, str]:
    """Environment variable overrides to pass through to the server process.

    Mirrors `launchd._get_env_overrides()` -- same two variables, same "only if set"
    behaviour -- kept as a separate copy here so this module doesn't reach into
    launchd.py's internals.
    """
    env = {}
    for var in ["VAS_CONFIG", "VAS_DATA_DIR"]:
        if val := os.environ.get(var):
            env[var] = val
    return env


def server_entry(bin_path: str) -> dict:
    """Build the `mcpServers` entry for our server.

    No secret goes here: Claude Desktop's config file is not treated as sensitive
    storage, so the Anthropic key is read by the server from the key file managed by
    `ensure_api_key_file()`/`voice_ai_summary.llm.api_key_path()` instead.
    """
    return {"command": bin_path, "args": [], "env": _get_env_overrides()}


def _set_aside(path: Path, reason: str, *, dry_run: bool) -> None:
    """Copy an unreadable config out of the way before we overwrite it."""
    backup_path = path.with_name(f"{path.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    if dry_run:
        print(f"existing config {reason}; would back it up to {backup_path}", file=sys.stderr)
        return
    shutil.copy2(path, backup_path)
    print(f"existing config {reason}; backed up to {backup_path}", file=sys.stderr)


def _load_existing(path: Path, *, dry_run: bool = False) -> dict:
    """Read the existing config, backing up and starting fresh if it is corrupt."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        _set_aside(path, "was not valid JSON", dry_run=dry_run)
        return {}
    if not isinstance(data, dict):
        _set_aside(path, "was not a JSON object", dry_run=dry_run)
        return {}
    return data


def install(bin_path: str, *, path: Path | None = None, dry_run: bool = False) -> Path:
    """Merge our server entry into Claude Desktop's config file.

    Preserves every other top-level key and every other `mcpServers` entry. Idempotent:
    running it again with the same (or a different) `bin_path` just updates our entry.
    """
    config_path = _config_path(path)
    data = _load_existing(config_path, dry_run=dry_run)

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_KEY] = server_entry(bin_path)
    data["mcpServers"] = servers

    if not dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        config_path.write_text(text, encoding="utf-8")

    return config_path


def uninstall(*, path: Path | None = None) -> bool:
    """Remove our entry from Claude Desktop's config file, leaving everything else.

    Returns whether anything was actually removed.
    """
    config_path = _config_path(path)
    data = _load_existing(config_path)

    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or SERVER_KEY not in servers:
        return False

    del servers[SERVER_KEY]
    data["mcpServers"] = servers
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def ensure_api_key_file(*, path: Path | None = None) -> str:
    """Write the Anthropic key from the environment to the key file if it isn't there yet.

    Never returns or logs the key itself, only a status describing what happened.
    """
    from .llm import api_key_path

    key_path = path if path is not None else api_key_path()
    if key_path.exists():
        return f"key file already exists at {key_path}"

    env_key = os.environ.get("VAS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not env_key:
        return (
            "no VAS_ANTHROPIC_API_KEY/ANTHROPIC_API_KEY found in the environment; "
            f"write the key to {key_path} yourself (chmod 600)"
        )

    key_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(env_key)
    return f"wrote key to {key_path} (mode 600)"


def default_bin(name: str = "vas-mcp") -> str:
    """Resolve the installed `vas-mcp` console script."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    return name
