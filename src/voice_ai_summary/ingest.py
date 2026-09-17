"""Inbox ingestion: recorder files → content-addressed store → `recordings` rows."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .audio import sha256_file
from .config import Config
from .db import transaction, utcnow_iso
from .security import ensure_private_dir

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(
    r"^(mac_mic|mac_system|file)_(?P<device>.+)_(?P<ts>\d{8}T\d{6}Z)$",
)
_DEFAULT_TZ = "+00:00"


def parse_inbox_name(path: Path) -> dict | None:
    """Parse `{source}_{device}_{YYYYMMDDTHHMMSSZ}` from a recorder filename."""
    match = _NAME_RE.match(Path(path).stem)
    if not match:
        return None
    ts = datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return {
        "source": match.group(1),
        "device_id": match.group("device"),
        "started_at_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _read_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _delete_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        sidecar.unlink()


def ingest_file(
    conn: sqlite3.Connection,
    cfg: Config,
    path: Path,
    *,
    source: str | None = None,
    started_at_utc: str | None = None,
    tz_offset: str = _DEFAULT_TZ,
    move: bool = True,
) -> int | None:
    """Ingest one recording file, moving/copying it into the content-addressed store."""
    path = Path(path)
    sha = sha256_file(path)

    existing = conn.execute("SELECT id FROM recordings WHERE sha256 = ?", (sha,)).fetchone()
    if existing is not None:
        if move and path.parent == cfg.paths.inbox:
            _delete_sidecar(path)
            path.unlink()
        return existing["id"]

    sidecar = _read_sidecar(path)
    parsed = parse_inbox_name(path) or {}

    resolved_source = source or sidecar.get("source") or parsed.get("source") or "file"
    resolved_device = sidecar.get("device_id") or parsed.get("device_id") or ""
    resolved_started = (
        started_at_utc
        or sidecar.get("started_at_utc")
        or parsed.get("started_at_utc")
        or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    resolved_tz = tz_offset if tz_offset != _DEFAULT_TZ else sidecar.get("tz_offset", tz_offset)

    started_dt = datetime.strptime(resolved_started, "%Y-%m-%dT%H:%M:%SZ")
    dest_rel = Path(started_dt.strftime("%Y/%m/%d")) / f"{sha[:16]}{path.suffix}"
    dest_abs = cfg.paths.store / dest_rel
    # Recorded speech lives under here, so the dated directory must not be
    # group/other-readable even if it predates `security.ensure_private_dir`.
    ensure_private_dir(dest_abs.parent)

    if move:
        shutil.move(str(path), str(dest_abs))
        _delete_sidecar(path)
    else:
        shutil.copy2(path, dest_abs)
    # `shutil.move`/`copy2` preserve the source file's mode, which may predate this
    # process's private umask (e.g. a file the recorder wrote before it applied one).
    os.chmod(dest_abs, 0o600)

    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO recordings
                (source, device_id, started_at_utc, tz_offset, sha256, storage_path,
                 original_name, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_source,
                resolved_device,
                resolved_started,
                resolved_tz,
                sha,
                dest_rel.as_posix(),
                path.name,
                utcnow_iso(),
            ),
        )
    return cur.lastrowid


def ingest_inbox(conn: sqlite3.Connection, cfg: Config) -> list[int]:
    """Ingest every non-sidecar file in the inbox, oldest first, skipping in-progress writes."""
    if not cfg.paths.inbox.is_dir():
        return []
    now = datetime.now(UTC).timestamp()
    # `.part` is the recorder's "still open" marker; mtime alone is not enough because
    # the encoder flushes to disk only every few seconds.
    candidates = [
        p
        for p in cfg.paths.inbox.iterdir()
        if p.is_file()
        and p.suffix not in (".json", ".part")
        and not p.name.startswith(".")
        and (now - p.stat().st_mtime) >= 5
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)

    ids: list[int] = []
    for path in candidates:
        try:
            rec_id = ingest_file(conn, cfg, path, move=True)
        except FileNotFoundError:
            # The recorder can delete/rotate a file between the `iterdir()` listing
            # above and this move (e.g. it also cleans up its own old files) - the
            # worker must not crash-loop over a race like that.
            logger.warning("skipping %s: vanished before it could be ingested", path)
            continue
        if rec_id is not None:
            ids.append(rec_id)
    return ids
