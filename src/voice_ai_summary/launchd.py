"""macOS launchd service installation and management."""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config

WORKER_LABEL = "com.voiceaisummary.worker"
DIGEST_LABEL = "com.voiceaisummary.digest"


def _program_args(vas_bin: str, *args: str) -> list[str]:
    """Run `vas` through a zsh login shell.

    A launchd agent starts with a bare environment: none of the secrets the README tells
    the user to export (`ANTHROPIC_API_KEY` and friends) are visible to it, so a digest
    run would fail on its first Claude call. zsh reads ~/.zshenv on every invocation,
    which is where those exports live.
    """
    return ["/bin/zsh", "-lc", shlex.join([vas_bin, *args])]


def render_worker_plist(vas_bin: str, log_dir: Path) -> str:
    """Render plist XML for the background worker service.

    Args:
        vas_bin: Path to the `vas` executable.
        log_dir: Directory for stdout/stderr logs.

    Returns:
        Plist XML string.
    """
    plist_dict = {
        "Label": WORKER_LABEL,
        "ProgramArguments": _program_args(vas_bin, "worker"),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_dir / f"{WORKER_LABEL}.log"),
        "StandardErrorPath": str(log_dir / f"{WORKER_LABEL}.err"),
        "ProcessType": "Background",
        "EnvironmentVariables": _get_env_overrides(),
    }
    return plistlib.dumps(plist_dict).decode("utf-8")


def render_digest_plist(vas_bin: str, log_dir: Path, hour: int, minute: int) -> str:
    """Render plist XML for the daily digest trigger.

    Args:
        vas_bin: Path to the `vas` executable.
        log_dir: Directory for stdout/stderr logs.
        hour: Hour of day (0-23) in local time.
        minute: Minute of hour (0-59).

    Returns:
        Plist XML string.
    """
    plist_dict = {
        "Label": DIGEST_LABEL,
        "ProgramArguments": _program_args(vas_bin, "digest", "--deliver"),
        "StartCalendarInterval": {
            "Hour": hour,
            "Minute": minute,
        },
        "RunAtLoad": False,
        "StandardOutPath": str(log_dir / f"{DIGEST_LABEL}.log"),
        "StandardErrorPath": str(log_dir / f"{DIGEST_LABEL}.err"),
        "EnvironmentVariables": _get_env_overrides(),
    }
    return plistlib.dumps(plist_dict).decode("utf-8")


def install(cfg: Config, *, vas_bin: str | None = None, dry_run: bool = False) -> list[Path]:
    """Install or update launchd services.

    Creates plist files in ~/Library/LaunchAgents and loads them via launchctl.
    On non-macOS, only writes plist files and prints launchctl commands.

    Args:
        cfg: Configuration with schedule settings.
        vas_bin: Path to vas executable (defaults to `which vas` or sys.argv[0]).
        dry_run: If True, only write files and print commands (don't run launchctl).

    Returns:
        List of plist file paths written.
    """
    if vas_bin is None:
        vas_bin = shutil.which("vas") or sys.argv[0]

    log_dir = Path.home() / "Library" / "Logs" / "VoiceAISummary"
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    agents_dir = Path.home() / "Library" / "LaunchAgents"
    if not dry_run:
        agents_dir.mkdir(parents=True, exist_ok=True)

    plist_paths: list[Path] = []

    # Write and load worker service
    worker_path = agents_dir / f"{WORKER_LABEL}.plist"
    worker_plist = render_worker_plist(vas_bin, log_dir)
    if not dry_run:
        worker_path.write_text(worker_plist, encoding="utf-8")
    plist_paths.append(worker_path)

    # Write and load digest service
    digest_path = agents_dir / f"{DIGEST_LABEL}.plist"
    digest_plist = render_digest_plist(
        vas_bin, log_dir, cfg.schedule.digest_hour, cfg.schedule.digest_minute
    )
    if not dry_run:
        digest_path.write_text(digest_plist, encoding="utf-8")
    plist_paths.append(digest_path)

    # Load services
    if sys.platform == "darwin":
        uid = os.getuid()
        for plist_path in plist_paths:
            _launchctl_bootout(uid, str(plist_path), dry_run=dry_run)
            _launchctl_bootstrap(uid, str(plist_path), dry_run=dry_run)
    else:
        # Print commands for non-macOS
        uid = os.getuid()
        for plist_path in plist_paths:
            print(f"launchctl bootout gui/{uid} {plist_path}")
            print(f"launchctl bootstrap gui/{uid} {plist_path}")

    return plist_paths


def uninstall(cfg: Config, *, vas_bin: str | None = None, dry_run: bool = False) -> None:
    """Uninstall launchd services.

    Boots out services and removes plist files.

    Args:
        cfg: Configuration (unused, for API consistency).
        vas_bin: Unused, for API consistency.
        dry_run: If True, only print commands.
    """
    agents_dir = Path.home() / "Library" / "LaunchAgents"

    if sys.platform == "darwin":
        uid = os.getuid()
        for label in [WORKER_LABEL, DIGEST_LABEL]:
            plist_path = agents_dir / f"{label}.plist"
            _launchctl_bootout(uid, str(plist_path), dry_run=dry_run)

    for label in [WORKER_LABEL, DIGEST_LABEL]:
        plist_path = agents_dir / f"{label}.plist"
        if not dry_run and plist_path.exists():
            plist_path.unlink()
        elif dry_run:
            print(f"rm {plist_path}")


def _launchctl_bootout(uid: int, plist_path: str, *, dry_run: bool = False) -> None:
    """Boot out (stop) a launchd service. Ignores failures."""
    cmd = ["launchctl", "bootout", f"gui/{uid}", plist_path]
    if dry_run:
        print(" ".join(cmd))
    else:
        subprocess.run(cmd, capture_output=True)  # Ignore errors


def _launchctl_bootstrap(uid: int, plist_path: str, *, dry_run: bool = False) -> None:
    """Bootstrap (start) a launchd service."""
    cmd = ["launchctl", "bootstrap", f"gui/{uid}", plist_path]
    if dry_run:
        print(" ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def _get_env_overrides() -> dict[str, str]:
    """Get environment variable overrides from current env if set."""
    env = {}
    for var in ["VAS_CONFIG", "VAS_DATA_DIR"]:
        if val := os.environ.get(var):
            env[var] = val
    return env
