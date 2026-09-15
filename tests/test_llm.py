"""Client construction and usage accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from voice_ai_summary import llm
from voice_ai_summary.config import Config


def test_make_client_prefers_dedicated_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-key")
    monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "vas-key")
    cfg = Config()
    cfg.paths.data_dir = tmp_path
    client = llm.make_client(cfg)
    assert client.api_key == "vas-key"

    monkeypatch.delenv("VAS_ANTHROPIC_API_KEY")
    assert llm.make_client(cfg).api_key == "shared-key"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="VAS_ANTHROPIC_API_KEY"):
        llm.make_client(cfg)


def test_track_and_summarize_usage(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    llm.set_usage_path(path)
    try:
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1_000_000,
                output_tokens=100_000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        )
        llm.track_usage("reduce", "claude-opus-5", resp)
        llm.track_usage("map", "claude-haiku-4-5", resp)
        llm.track_usage("map", "claude-haiku-4-5", SimpleNamespace(usage=None))
    finally:
        llm.set_usage_path(None)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["purpose"] == "reduce"

    rows = llm.summarize_usage(path, days=1)
    assert [(r["purpose"], r["calls"]) for r in rows] == [("reduce", 1), ("map", 1)]
    assert rows[0]["usd"] == pytest.approx(5.0 + 2.5)  # 1M in @ $5 + 100k out @ $25
    assert rows[1]["usd"] == pytest.approx(1.0 + 0.5)


def test_track_usage_is_noop_without_path(tmp_path) -> None:
    llm.set_usage_path(None)
    llm.track_usage("map", "claude-haiku-4-5", SimpleNamespace(usage=SimpleNamespace()))
    assert not (tmp_path / "usage.jsonl").exists()
