"""Commit and push the daily digest to a private git repository.

Lets a Claude session read the digest via GitHub, since Claude can't reach the user's
Mac directly. The digest may contain other people's speech and client names, so the
target repo must be private.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from ..config import DeliverConfig

log = logging.getLogger(__name__)

_Runner = Callable[..., subprocess.CompletedProcess]


def publish_to_repo(
    cfg: DeliverConfig,
    day: str,
    markdown: str,
    *,
    runner: _Runner | None = None,
) -> str:
    """Write `<repo_path>/<repo_subdir>/<day>.md` and commit + push it.

    Returns the path committed, relative to the repo root. Re-running for a day that was
    already committed with the same content is a no-op past the write (nothing to commit).
    Raises RuntimeError if `repo_path` is not a valid git checkout, or if add/commit/push fail.
    """
    if runner is None:
        runner = subprocess.run
    repo = _resolve_repo_path(cfg.repo_path)

    if cfg.repo_branch:
        _run(runner, repo, ["checkout", cfg.repo_branch])

    pull_target = cfg.repo_branch or "HEAD"
    pull = _run(runner, repo, ["pull", "--ff-only", cfg.repo_remote, pull_target], check=False)
    if pull.returncode != 0:
        log.warning("git pull failed for %s, continuing offline: %s", repo, pull.stderr.strip())

    rel_path = Path(cfg.repo_subdir) / f"{day}.md"
    abs_path = repo / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(markdown, encoding="utf-8")

    _run(runner, repo, ["add", str(rel_path)])

    diff = _run(runner, repo, ["diff", "--cached", "--quiet", "--", str(rel_path)], check=False)
    if diff.returncode != 0:
        _run(runner, repo, ["commit", "-m", f"Add digest for {day}", "--", str(rel_path)])
        push_target = f"HEAD:{cfg.repo_branch}" if cfg.repo_branch else "HEAD"
        _run(runner, repo, ["push", cfg.repo_remote, push_target])

    return str(rel_path)


def _resolve_repo_path(repo_path: str) -> Path:
    if not repo_path:
        raise RuntimeError("repo_path not configured in [deliver] section.")
    repo = Path(repo_path).expanduser()
    if not repo.is_dir():
        raise RuntimeError(f"repo_path {repo} does not exist or is not a directory.")
    if not (repo / ".git").exists():
        raise RuntimeError(f"repo_path {repo} is not a git repository (no .git).")
    return repo


def _run(
    runner: _Runner,
    repo: Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run `git -C <repo> <args>`, raising RuntimeError on failure when `check`."""
    result = runner(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result
