"""Tests for the process-umask, private-file and secret-redaction helpers."""

from __future__ import annotations

import json
import os
import stat
import sys
from types import SimpleNamespace

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


class TestHarden:
    def test_fixes_permissive_files_and_dirs_under_root(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        root = cfg.paths.root
        sub = root / "sub"
        sub.mkdir(parents=True, mode=0o755)
        root.chmod(0o755)
        f1 = root / "a.txt"
        f1.write_text("x", encoding="utf-8")
        f1.chmod(0o644)
        f2 = sub / "b.txt"
        f2.write_text("y", encoding="utf-8")
        f2.chmod(0o644)

        changes = security.harden(cfg, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key")

        assert _mode(root) == 0o700
        assert _mode(sub) == 0o700
        assert _mode(f1) == 0o600
        assert _mode(f2) == 0o600
        changed_paths = {c.path for c in changes}
        assert changed_paths == {root, sub, f1, f2}
        for c in changes:
            assert c.new_mode in (0o700, 0o600)

    def test_dry_run_reports_without_changing(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        root = cfg.paths.root
        root.mkdir(parents=True, mode=0o755)
        f1 = root / "a.txt"
        f1.write_text("x", encoding="utf-8")
        f1.chmod(0o644)

        changes = security.harden(
            cfg, dry_run=True, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key"
        )

        assert _mode(root) == 0o755
        assert _mode(f1) == 0o644
        changed_paths = {c.path for c in changes}
        assert changed_paths == {root, f1}

    def test_already_private_tree_reports_nothing(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        cfg.ensure_dirs()

        changes = security.harden(cfg, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key")

        assert changes == []

    def test_skips_symlinks(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        root = cfg.paths.root
        root.mkdir(parents=True, mode=0o700)
        target = tmp_path / "outside.txt"
        target.write_text("z", encoding="utf-8")
        target.chmod(0o644)
        link = root / "link.txt"
        link.symlink_to(target)

        changes = security.harden(cfg, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key")

        assert changes == []
        assert _mode(target) == 0o644  # untouched: the symlink was never followed

    def test_hardens_mirror_dir_and_its_markdown_files_only(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        cfg.paths.digest_mirror_dir = tmp_path / "mirror"
        mirror = cfg.paths.digest_mirror
        mirror.mkdir(parents=True, mode=0o755)
        (tmp_path).chmod(0o755)  # mirror's parent must be left alone
        md = mirror / "2026-09-17.md"
        md.write_text("# digest", encoding="utf-8")
        md.chmod(0o644)
        other = mirror / "notes.txt"
        other.write_text("ignored", encoding="utf-8")
        other.chmod(0o644)

        changes = security.harden(
            cfg, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key-dir" / "no-key"
        )

        assert _mode(mirror) == 0o700
        assert _mode(md) == 0o600
        assert _mode(other) == 0o644  # not a *.md file: left alone
        assert _mode(tmp_path) == 0o755  # mirror's parent is not touched
        changed_paths = {c.path for c in changes}
        assert md in changed_paths
        assert mirror in changed_paths
        assert other not in changed_paths

    def test_hardens_key_file_and_its_parent_dir(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        key_dir = tmp_path / "keydir"
        key_dir.mkdir(mode=0o755)
        key_path = key_dir / "anthropic_api_key"
        key_path.write_text("secret", encoding="utf-8")
        key_path.chmod(0o644)

        changes = security.harden(cfg, log_dir=tmp_path / "no-logs", key_path=key_path)

        assert _mode(key_dir) == 0o700
        assert _mode(key_path) == 0o600
        changed_paths = {c.path for c in changes}
        assert {key_dir, key_path} <= changed_paths

    def test_hardens_log_dir_and_its_files(self, tmp_path) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        logs = tmp_path / "logs"
        logs.mkdir(mode=0o755)
        log_file = logs / "worker.log"
        log_file.write_text("hi", encoding="utf-8")
        log_file.chmod(0o644)

        changes = security.harden(cfg, log_dir=logs, key_path=tmp_path / "no-key")

        assert _mode(logs) == 0o700
        assert _mode(log_file) == 0o600
        changed_paths = {c.path for c in changes}
        assert {logs, log_file} <= changed_paths

    def test_never_raises_on_a_failed_chmod(self, tmp_path, monkeypatch) -> None:
        from voice_ai_summary.config import Config

        cfg = Config()
        cfg.paths.data_dir = tmp_path / "data"
        root = cfg.paths.root
        root.mkdir(parents=True, mode=0o755)
        f = root / "a.txt"
        f.write_text("x", encoding="utf-8")
        f.chmod(0o644)

        def _boom(self, mode):
            raise OSError("nope")

        monkeypatch.setattr("pathlib.Path.chmod", _boom)

        changes = security.harden(cfg, log_dir=tmp_path / "no-logs", key_path=tmp_path / "no-key")
        assert changes == []  # every chmod failed, so nothing to report as changed


class TestFilevaultStatus:
    def test_non_darwin_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(security.sys, "platform", "linux")
        assert security.filevault_status() is None

    def test_on(self, monkeypatch) -> None:
        monkeypatch.setattr(security.sys, "platform", "darwin")
        monkeypatch.setattr(
            security.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="FileVault is On.\n", stderr=""),
        )
        assert security.filevault_status() == "on"

    def test_off(self, monkeypatch) -> None:
        monkeypatch.setattr(security.sys, "platform", "darwin")
        monkeypatch.setattr(
            security.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="FileVault is Off.\n", stderr=""),
        )
        assert security.filevault_status() == "off"

    def test_garbage_output_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(security.sys, "platform", "darwin")
        monkeypatch.setattr(
            security.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout="unexpected\n", stderr=""),
        )
        assert security.filevault_status() is None

    def test_failure_to_run_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(security.sys, "platform", "darwin")

        def _raise(*a, **k):
            raise OSError("fdesetup not found")

        monkeypatch.setattr(security.subprocess, "run", _raise)
        assert security.filevault_status() is None


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
