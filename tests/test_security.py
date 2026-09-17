"""Tests for the process-umask, private-file and secret-redaction helpers."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from voice_ai_summary import security

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def restore_umask():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


class TestApplyProcessUmask:
    def test_returns_previous_mask_and_makes_new_files_private(
        self, tmp_path, restore_umask
    ) -> None:
        previous = os.umask(0o022)  # known starting point, restored by the fixture
        os.umask(previous)

        old = security.apply_process_umask()
        assert old == 0o022

        path = tmp_path / "created-after-umask"
        path.write_text("x", encoding="utf-8")
        assert _mode(path) == 0o600


class TestOpenPrivate:
    def test_creates_file_at_mode_600(self, tmp_path) -> None:
        path = tmp_path / "secret.txt"
        with security.open_private(path, "w") as f:
            f.write("hello")
        assert path.read_text(encoding="utf-8") == "hello"
        assert _mode(path) == 0o600

    def test_append_mode_appends(self, tmp_path) -> None:
        path = tmp_path / "log.jsonl"
        with security.open_private(path, "a") as f:
            f.write("one\n")
        with security.open_private(path, "a") as f:
            f.write("two\n")
        assert path.read_text(encoding="utf-8") == "one\ntwo\n"
        assert _mode(path) == 0o600

    def test_rejects_other_modes(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="'w' or 'a'"):
            security.open_private(tmp_path / "x", "r")


class TestEnsurePrivateDir:
    def test_fixes_permissive_dir(self, tmp_path) -> None:
        d = tmp_path / "data"
        d.mkdir(mode=0o755)
        assert security.ensure_private_dir(d) is True
        assert _mode(d) == 0o700

    def test_already_private_dir_returns_false(self, tmp_path) -> None:
        d = tmp_path / "data"
        d.mkdir(mode=0o700)
        assert security.ensure_private_dir(d) is False
        assert _mode(d) == 0o700

    def test_creates_missing_dir_with_parents(self, tmp_path) -> None:
        d = tmp_path / "a" / "b" / "c"
        security.ensure_private_dir(d)
        assert d.is_dir()


class TestEnsurePrivateFile:
    def test_fixes_permissive_file(self, tmp_path) -> None:
        f = tmp_path / "key"
        f.write_text("x", encoding="utf-8")
        f.chmod(0o644)
        assert security.ensure_private_file(f) is True
        assert _mode(f) == 0o600

    def test_missing_file_returns_false(self, tmp_path) -> None:
        assert security.ensure_private_file(tmp_path / "nope") is False

    def test_already_private_file_returns_false(self, tmp_path) -> None:
        f = tmp_path / "key"
        f.write_text("x", encoding="utf-8")
        f.chmod(0o600)
        assert security.ensure_private_file(f) is False


class TestRedact:
    def test_anthropic_key(self) -> None:
        out = security.redact("key is sk-ant-abcdEFGH12345")
        assert "sk-ant-abcdEFGH12345" not in out
        assert "sk-ant-…[redacted]" in out

    def test_slack_token(self) -> None:
        out = security.redact("posting with xoxb-1234-5678-abcdEFGH for auth")
        assert "xoxb-1234-5678-abcdEFGH" not in out
        assert "xox*-[redacted]" in out

    def test_credentials_in_url(self) -> None:
        out = security.redact("connect to https://alice:s3cr3t@example.com/db")
        assert "alice:s3cr3t@" not in out
        assert out == "connect to https://***:***@example.com/db"

    def test_slack_webhook(self) -> None:
        out = security.redact("post to https://hooks.slack.com/services/T000/B000/XXXXXXXXXXXX")
        assert "T000/B000/XXXXXXXXXXXX" not in out
        assert "hooks.slack.com/services/[redacted]" in out

    def test_generic_key_value(self) -> None:
        out = security.redact("api_key=abc123 password: hunter2")
        assert "abc123" not in out
        assert "hunter2" not in out
        assert "api_key=[redacted]" in out
        assert "password=[redacted]" in out

    def test_plain_url_untouched(self) -> None:
        text = "see https://example.com/path for details"
        assert security.redact(text) == text

    def test_japanese_prose_without_value_untouched(self) -> None:
        text = "トーコンを更新しました"
        assert security.redact(text) == text

    def test_masks_only_the_secret_within_a_longer_line(self) -> None:
        text = "starting up: token=abc123 and continuing normally"
        out = security.redact(text)
        assert "abc123" not in out
        assert out.startswith("starting up: token=[redacted]")
        assert out.endswith("and continuing normally")


class TestAudit:
    def test_writes_one_parseable_private_json_line(self, tmp_path) -> None:
        security.audit(tmp_path, "summarize", {"episode": "42"}, "ok")

        path = tmp_path / security.AUDIT_FILENAME
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert set(rec) == {"ts", "tool", "args", "outcome"}
        assert rec["tool"] == "summarize"
        assert rec["args"] == {"episode": "42"}
        assert rec["outcome"] == "ok"
        assert _mode(path) == 0o600

    def test_redacts_secrets_in_outcome_and_args(self, tmp_path) -> None:
        security.audit(
            tmp_path,
            "deliver",
            {"note": "leaked sk-ant-abcdEFGH12345", "count": 3},
            "failed with sk-ant-zzzzZZZZ99999 in the response",
        )
        rec = json.loads((tmp_path / security.AUDIT_FILENAME).read_text(encoding="utf-8"))
        assert "sk-ant-zzzzZZZZ99999" not in rec["outcome"]
        assert "sk-ant-abcdEFGH12345" not in rec["args"]["note"]
        assert rec["args"]["count"] == 3

    def test_outcome_capped_at_300_chars(self, tmp_path) -> None:
        security.audit(tmp_path, "tool", {}, "x" * 1000)
        rec = json.loads((tmp_path / security.AUDIT_FILENAME).read_text(encoding="utf-8"))
        assert len(rec["outcome"]) == 300

    def test_unwritable_root_does_not_raise(self, tmp_path) -> None:
        not_a_dir = tmp_path / "not-a-dir"
        not_a_dir.write_text("x", encoding="utf-8")
        security.audit(not_a_dir, "tool", {}, "ok")  # must not raise


def test_ensure_dirs_creates_private_directories(tmp_path, monkeypatch) -> None:
    """Every data directory is recorded speech; none may be listable by other accounts."""
    import stat

    from voice_ai_summary.config import Config

    cfg = Config()
    cfg.paths.data_dir = tmp_path / "data"
    cfg.paths.digest_mirror_dir = tmp_path / "mirror"
    monkeypatch.setattr("os.umask", lambda _m: 0o022)  # a permissive umask must not matter

    cfg.ensure_dirs()

    for d in (
        cfg.paths.root,
        cfg.paths.inbox,
        cfg.paths.store,
        cfg.paths.digests,
        tmp_path / "mirror",
    ):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
