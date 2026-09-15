"""Shared pytest fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from voice_ai_summary.config import Config


@pytest.fixture
def vas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Config, sqlite3.Connection]]:
    """A fresh config + connected DB rooted at a tmp dir, with the fake ASR backend."""
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VAS_ASR_BACKEND", "fake")

    from voice_ai_summary.config import load_config
    from voice_ai_summary.db import connect

    cfg = load_config()
    cfg.ensure_dirs()
    conn = connect(cfg.paths.db_path)
    yield cfg, conn
    conn.close()
