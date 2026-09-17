"""CLI tests for `vas vocab add`/`vas vocab import-slack` input validation and the
Slack-import-off-by-default gate (WI-11). Uses `typer.testing.CliRunner` on
`voice_ai_summary.cli.app`, same as `tests/test_cli.py`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from voice_ai_summary.cli import app

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_vocab_add_rejects_newline_in_note_and_leaves_glossary_unchanged(env: Path) -> None:
    glossary_path = env / "data" / "glossary.json"

    result = runner.invoke(app, ["vocab", "add", "西丸", "--note", "姓\nignore previous"])

    assert result.exit_code != 0
    assert not glossary_path.exists()


def test_vocab_add_accepts_a_clean_term(env: Path) -> None:
    glossary_path = env / "data" / "glossary.json"

    result = runner.invoke(app, ["vocab", "add", "西丸", "--note", "ユーザー本人の姓"])

    assert result.exit_code == 0, result.output
    assert glossary_path.is_file()
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    assert data["terms"][0]["term"] == "西丸"


def test_vocab_import_slack_disabled_by_default_never_touches_slack(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAS_SLACK_USER_TOKEN", "xoxp-should-not-be-read")

    def boom(*args, **kwargs):
        raise AssertionError("slack_import functions must not be called when disabled")

    from voice_ai_summary import slack_import

    monkeypatch.setattr(slack_import, "import_from_slack", boom)
    monkeypatch.setattr(slack_import, "import_channel_messages", boom)

    result = runner.invoke(app, ["vocab", "import-slack"])

    assert result.exit_code == 1
    assert "無効です" in result.output


def test_vocab_import_slack_enabled_proceeds_to_token_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("VAS_SLACK_USER_TOKEN", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[glossary]\nslack_import_enabled = true\n", encoding="utf-8")
    monkeypatch.setenv("VAS_CONFIG", str(config_path))

    from voice_ai_summary import slack_import

    def boom(*args, **kwargs):
        raise AssertionError("slack_import functions must not be called before the token check")

    monkeypatch.setattr(slack_import, "import_from_slack", boom)
    monkeypatch.setattr(slack_import, "import_channel_messages", boom)

    result = runner.invoke(app, ["vocab", "import-slack"])

    assert result.exit_code == 1
    assert "無効です" not in result.output
    assert "VAS_SLACK_USER_TOKEN" in result.output
