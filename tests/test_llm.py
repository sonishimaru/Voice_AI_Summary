"""Client construction and usage accounting."""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from voice_ai_summary import llm
from voice_ai_summary.config import Config


def test_make_client_prefers_dedicated_key(monkeypatch, tmp_path) -> None:
    # Point VAS_CONFIG into tmp_path so `api_key_path()` resolves to a file that does
    # not exist. Without this the test reads whatever key file the machine running it
    # happens to have -- `vas install-desktop` writes one -- and the no-key case then
    # passes here and fails on a real install.
    monkeypatch.setenv("VAS_CONFIG", str(tmp_path / "config.toml"))
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


def test_api_key_falls_back_to_key_file(monkeypatch, tmp_path) -> None:
    """Claude Desktop starts the MCP server with a bare environment, so the key file
    next to config.toml (`llm.api_key_path()`) must work with no env vars set."""
    monkeypatch.delenv("VAS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("VAS_CONFIG", str(config_path))

    assert llm.api_key_path() == config_path.parent / llm.API_KEY_FILENAME
    assert llm.api_key() is None  # no file yet

    key_path = llm.api_key_path()
    key_path.write_text("  file-key  \n", encoding="utf-8")
    assert llm.api_key() == "file-key"

    # An env var still takes priority over the file.
    monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "env-key")
    assert llm.api_key() == "env-key"


def test_api_key_repairs_a_permissive_key_file(monkeypatch, tmp_path, capsys) -> None:
    """A key file with any group/world bits set is repaired to 0600, a one-line
    warning is printed, and the key is still returned -- refusing here would take the
    MCP server down right after `update_app` created the file at the wrong mode."""
    monkeypatch.delenv("VAS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("VAS_CONFIG", str(config_path))

    key_path = llm.api_key_path()
    key_path.write_text("sk-test-key", encoding="utf-8")
    key_path.chmod(0o644)

    result = llm.api_key()

    assert result == "sk-test-key"
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    err = capsys.readouterr().err
    assert "0600" in err


def test_api_key_raises_when_the_chmod_repair_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("VAS_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("VAS_CONFIG", str(config_path))

    key_path = llm.api_key_path()
    key_path.write_text("sk-test-key", encoding="utf-8")
    key_path.chmod(0o644)

    def _boom(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(os, "chmod", _boom)

    with pytest.raises(RuntimeError, match=str(key_path)):
        llm.api_key()


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


def test_track_usage_creates_usage_file_at_mode_600(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    llm.set_usage_path(path)
    try:
        resp = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        )
        llm.track_usage("map", "claude-haiku-4-5", resp)
    finally:
        llm.set_usage_path(None)

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_track_usage_is_noop_without_path(tmp_path) -> None:
    llm.set_usage_path(None)
    llm.track_usage("map", "claude-haiku-4-5", SimpleNamespace(usage=SimpleNamespace()))
    assert not (tmp_path / "usage.jsonl").exists()


def test_daily_budget_stops_further_calls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "k")
    cfg = Config()
    cfg.paths.data_dir = tmp_path
    cfg.llm.daily_budget_usd = 1.0
    llm.make_client(cfg)
    big = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=300_000,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    )
    try:
        # Recording never raises: a response that was paid for is always returned.
        llm.check_budget()
        llm.track_usage("reduce", "claude-opus-5", big)  # $1.50 > $1.00
        assert llm.spent_today(tmp_path / "usage.jsonl") == pytest.approx(1.5)

        # The *next* call, and the next client, are what stop.
        with pytest.raises(llm.BudgetExceeded):
            llm.check_budget()
        with pytest.raises(llm.BudgetExceeded):
            llm.make_client(cfg)

        cfg.llm.daily_budget_usd = 10.0
        llm.make_client(cfg)
        llm.check_budget()
    finally:
        llm.set_usage_path(None)


def test_unknown_model_is_priced_at_the_top_of_the_table(tmp_path) -> None:
    """An unrecognised model must not silently disable the daily cap."""
    llm.set_usage_path(tmp_path / "usage.jsonl", daily_budget_usd=1.0)
    try:
        llm.track_usage(
            "map",
            "claude-something-unreleased",
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1_000_000,
                    output_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                )
            ),
        )
        assert llm.spent_today(tmp_path / "usage.jsonl") == pytest.approx(
            max(llm.PRICES.values())[0]
        )
        with pytest.raises(llm.BudgetExceeded):
            llm.check_budget()
    finally:
        llm.set_usage_path(None)


def test_friendly_api_error_maps_setup_problems() -> None:
    def err(status: int, message: str) -> anthropic.APIStatusError:
        response = httpx.Response(
            status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        return anthropic.APIStatusError(message, response=response, body=None)

    low = llm.friendly_api_error(err(400, "Your credit balance is too low"), "m")
    assert low is not None and "billing" in str(low)
    assert "VAS_ANTHROPIC_API_KEY" in str(llm.friendly_api_error(err(401, "bad key"), "m"))
    assert "config.toml" in str(llm.friendly_api_error(err(404, "no model"), "claude-x"))
    # A genuine server fault stays an exception the caller must deal with.
    assert llm.friendly_api_error(err(500, "overloaded"), "m") is None


class _Response:
    """The shape `messages.parse` returns, as far as `parsed_or_raise` reads it."""

    def __init__(self, *, stop_reason: str, parsed_output=None, stop_details=None) -> None:
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output
        self.stop_details = stop_details


class _StopDetails:
    def __init__(self, category: str, explanation: str = "") -> None:
        self.category = category
        self.explanation = explanation


def test_parsed_or_raise_returns_the_parsed_output() -> None:
    response = _Response(stop_reason="end_turn", parsed_output={"fixes": []})
    assert llm.parsed_or_raise(response, purpose="correction") == {"fixes": []}


def test_a_refusal_names_the_category_and_is_not_retryable() -> None:
    """A refusal must not look like truncation: retrying a smaller input cannot help."""
    response = _Response(
        stop_reason="refusal",
        stop_details=_StopDetails("cyber", "This request was declined."),
    )
    with pytest.raises(RuntimeError) as excinfo:
        llm.parsed_or_raise(response, purpose="correction")
    assert not isinstance(excinfo.value, llm.OutputTruncated)
    assert "cyber" in str(excinfo.value)
    assert "declined" in str(excinfo.value)


def test_a_refusal_without_details_still_reports_something_useful() -> None:
    """`stop_details` is only populated for refusals, and even then may be sparse."""
    response = _Response(stop_reason="refusal")
    with pytest.raises(RuntimeError, match="unspecified"):
        llm.parsed_or_raise(response, purpose="correction")


def test_no_parsed_output_is_reported_as_truncation() -> None:
    """`messages.parse` returns None rather than raising; callers split and retry."""
    response = _Response(stop_reason="max_tokens", parsed_output=None)
    with pytest.raises(llm.OutputTruncated, match="max_tokens"):
        llm.parsed_or_raise(response, purpose="glossary extraction")
