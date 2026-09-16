"""Tests for the Claude Desktop MCP server registration."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from voice_ai_summary.desktop import (
    SERVER_KEY,
    ensure_api_key_file,
    install,
    server_entry,
    uninstall,
)


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep VAS_CONFIG/VAS_DATA_DIR from the real environment out of these tests."""
    monkeypatch.delenv("VAS_CONFIG", raising=False)
    monkeypatch.delenv("VAS_DATA_DIR", raising=False)


class TestInstall:
    def test_install_creates_missing_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "nested" / "claude_desktop_config.json"

        result = install("/usr/local/bin/vas-mcp", path=config_path)

        assert result == config_path
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["mcpServers"][SERVER_KEY]["command"] == "/usr/local/bin/vas-mcp"
        assert data["mcpServers"][SERVER_KEY]["args"] == []

    def test_install_preserves_other_servers_and_top_level_keys(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {"other-tool": {"command": "/bin/other", "args": ["--flag"]}},
                    "someOtherSetting": True,
                }
            ),
            encoding="utf-8",
        )

        install("/usr/local/bin/vas-mcp", path=config_path)

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["someOtherSetting"] is True
        assert data["mcpServers"]["other-tool"] == {"command": "/bin/other", "args": ["--flag"]}
        assert data["mcpServers"][SERVER_KEY]["command"] == "/usr/local/bin/vas-mcp"

    def test_install_twice_is_idempotent_and_updates_command(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"

        install("/usr/local/bin/vas-mcp", path=config_path)
        install("/opt/homebrew/bin/vas-mcp", path=config_path)

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert len(data["mcpServers"]) == 1
        assert data["mcpServers"][SERVER_KEY]["command"] == "/opt/homebrew/bin/vas-mcp"

    def test_install_backs_up_corrupt_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text("{not valid json", encoding="utf-8")

        install("/usr/local/bin/vas-mcp", path=config_path)

        # Original corrupt content is preserved in a backup, not silently dropped.
        backups = list(tmp_path.glob("claude_desktop_config.json.bak-*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "{not valid json"

        # A fresh, valid config with our entry was written in its place.
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["mcpServers"][SERVER_KEY]["command"] == "/usr/local/bin/vas-mcp"

    def test_install_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"

        result = install("/usr/local/bin/vas-mcp", path=config_path, dry_run=True)

        assert result == config_path
        assert not config_path.exists()

    def test_server_entry_has_no_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "sk-super-secret")
        entry = server_entry("/usr/local/bin/vas-mcp")
        assert "sk-super-secret" not in json.dumps(entry)

    def test_server_entry_passes_through_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VAS_CONFIG", "/etc/vas.toml")
        monkeypatch.setenv("VAS_DATA_DIR", "/data")
        entry = server_entry("/usr/local/bin/vas-mcp")
        assert entry["env"] == {"VAS_CONFIG": "/etc/vas.toml", "VAS_DATA_DIR": "/data"}


class TestUninstall:
    def test_uninstall_removes_only_our_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"
        install("/usr/local/bin/vas-mcp", path=config_path)
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        SERVER_KEY: {"command": "/usr/local/bin/vas-mcp", "args": [], "env": {}},
                        "other-tool": {"command": "/bin/other"},
                    }
                }
            ),
            encoding="utf-8",
        )

        removed = uninstall(path=config_path)

        assert removed is True
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert SERVER_KEY not in data["mcpServers"]
        assert "other-tool" in data["mcpServers"]

    def test_uninstall_returns_false_when_nothing_there(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"

        removed = uninstall(path=config_path)

        assert removed is False
        assert not config_path.exists()

    def test_uninstall_returns_false_on_missing_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text(json.dumps({"mcpServers": {"other-tool": {}}}), encoding="utf-8")

        removed = uninstall(path=config_path)

        assert removed is False
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["mcpServers"] == {"other-tool": {}}


class TestEnsureApiKeyFile:
    def test_writes_key_from_env_with_mode_600(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_path = tmp_path / "anthropic_api_key"
        monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "sk-test-key")

        status = ensure_api_key_file(path=key_path)

        assert key_path.exists()
        assert key_path.read_text(encoding="utf-8") == "sk-test-key"
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        assert mode == 0o600
        assert "sk-test-key" not in status

    def test_does_not_overwrite_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_path = tmp_path / "anthropic_api_key"
        key_path.write_text("existing-key", encoding="utf-8")
        monkeypatch.setenv("VAS_ANTHROPIC_API_KEY", "sk-new-key")

        status = ensure_api_key_file(path=key_path)

        assert key_path.read_text(encoding="utf-8") == "existing-key"
        assert "already exists" in status

    def test_reports_missing_key_without_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_path = tmp_path / "anthropic_api_key"
        monkeypatch.delenv("VAS_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        status = ensure_api_key_file(path=key_path)

        assert not key_path.exists()
        assert "no" in status.lower()
