"""Tests for launchd service installation and plist rendering."""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voice_ai_summary.config import Config
from voice_ai_summary.launchd import (
    DIGEST_LABEL,
    WORKER_LABEL,
    install,
    render_digest_plist,
    render_worker_plist,
    uninstall,
)


@pytest.fixture(autouse=True)
def no_real_launchctl():
    """Keep the test run out of the developer's own launchd domain.

    `install`/`uninstall` shell out to `launchctl bootstrap gui/<uid> <plist>` on macOS,
    and the tests below call them with `dry_run=False`. Without this, a test run
    registers services pointing at pytest tmp directories, which then outlive the run
    and fight with a real `vas install-launchd`.
    """
    with patch("subprocess.run") as mock_run:
        yield mock_run


class TestPlistRendering:
    """Test plist XML generation."""

    def test_render_worker_plist_structure(self, tmp_path: Path) -> None:
        """Worker plist should have correct structure."""
        log_dir = tmp_path / "logs"
        plist_xml = render_worker_plist("/usr/local/bin/vas", log_dir)

        # Parse the plist
        plist = plistlib.loads(plist_xml.encode("utf-8"))

        # Verify required keys
        assert plist["Label"] == WORKER_LABEL
        assert plist["ProgramArguments"] == ["/bin/zsh", "-lc", "/usr/local/bin/vas worker"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert plist["ProcessType"] == "Background"
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist

    def test_render_worker_plist_paths(self, tmp_path: Path) -> None:
        """Worker plist paths should reference log directory."""
        log_dir = tmp_path / "logs"
        plist_xml = render_worker_plist("/usr/local/bin/vas", log_dir)
        plist = plistlib.loads(plist_xml.encode("utf-8"))

        assert str(log_dir) in plist["StandardOutPath"]
        assert str(log_dir) in plist["StandardErrorPath"]

    def test_render_digest_plist_structure(self, tmp_path: Path) -> None:
        """Digest plist should have correct structure with calendar interval."""
        log_dir = tmp_path / "logs"
        plist_xml = render_digest_plist("/usr/local/bin/vas", log_dir, 22, 30)

        plist = plistlib.loads(plist_xml.encode("utf-8"))

        assert plist["Label"] == DIGEST_LABEL
        assert plist["ProgramArguments"] == [
            "/bin/zsh",
            "-lc",
            "/usr/local/bin/vas digest --deliver",
        ]
        assert plist["RunAtLoad"] is False
        assert "StartCalendarInterval" in plist
        assert plist["StartCalendarInterval"]["Hour"] == 22
        assert plist["StartCalendarInterval"]["Minute"] == 30

    def test_render_digest_plist_without_delivery(self, tmp_path: Path) -> None:
        """With no channel configured the job only writes the digest file: asking it to
        deliver would exit non-zero every night."""
        plist_xml = render_digest_plist("/usr/local/bin/vas", tmp_path, 22, 0, deliver=False)
        plist = plistlib.loads(plist_xml.encode("utf-8"))

        assert plist["ProgramArguments"] == ["/bin/zsh", "-lc", "/usr/local/bin/vas digest"]

    def test_install_drops_deliver_when_no_channel_enabled(self, tmp_path: Path) -> None:
        cfg = Config()
        # `notify` defaults to True, so it must be turned off too for a genuine
        # "nothing enabled" case.
        cfg.deliver.notify = False
        assert not any([cfg.deliver.slack, cfg.deliver.email, cfg.deliver.repo, cfg.deliver.notify])

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        digest_plist = plistlib.loads(
            (tmp_path / "Library" / "LaunchAgents" / f"{DIGEST_LABEL}.plist").read_bytes()
        )
        assert "--deliver" not in digest_plist["ProgramArguments"][2]

        cfg.deliver.slack = True
        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        digest_plist = plistlib.loads(
            (tmp_path / "Library" / "LaunchAgents" / f"{DIGEST_LABEL}.plist").read_bytes()
        )
        assert "--deliver" in digest_plist["ProgramArguments"][2]

    def test_install_keeps_deliver_for_notify_only(self, tmp_path: Path) -> None:
        """Regression: `notify` (a local macOS notification) defaults to True and is
        often the only delivery channel configured. `install` used to compute
        `deliver=cfg.deliver.slack or cfg.deliver.email`, which ignored `notify` (and
        `repo`) entirely, so the nightly digest job was installed without `--deliver`
        and the notification never fired."""
        cfg = Config()
        cfg.deliver.slack = False
        cfg.deliver.email = False
        cfg.deliver.repo = False
        cfg.deliver.notify = True

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        digest_plist = plistlib.loads(
            (tmp_path / "Library" / "LaunchAgents" / f"{DIGEST_LABEL}.plist").read_bytes()
        )
        assert "--deliver" in digest_plist["ProgramArguments"][2]

    def test_render_digest_plist_custom_time(self, tmp_path: Path) -> None:
        """Digest plist should use custom hour and minute."""
        log_dir = tmp_path / "logs"
        plist_xml = render_digest_plist("/usr/local/bin/vas", log_dir, 8, 15)

        plist = plistlib.loads(plist_xml.encode("utf-8"))

        assert plist["StartCalendarInterval"]["Hour"] == 8
        assert plist["StartCalendarInterval"]["Minute"] == 15

    def test_plist_env_vars_passthrough(self, tmp_path: Path) -> None:
        """Plist should pass through VAS_CONFIG and VAS_DATA_DIR if set."""
        log_dir = tmp_path / "logs"

        with patch.dict(os.environ, {"VAS_CONFIG": "/etc/vas.toml", "VAS_DATA_DIR": "/data"}):
            plist_xml = render_worker_plist("/usr/local/bin/vas", log_dir)
            plist = plistlib.loads(plist_xml.encode("utf-8"))

            assert plist["EnvironmentVariables"]["VAS_CONFIG"] == "/etc/vas.toml"
            assert plist["EnvironmentVariables"]["VAS_DATA_DIR"] == "/data"

    def test_plist_env_vars_not_set(self, tmp_path: Path) -> None:
        """Plist should not include unset env vars."""
        log_dir = tmp_path / "logs"

        with patch.dict(os.environ, {}, clear=True):
            plist_xml = render_worker_plist("/usr/local/bin/vas", log_dir)
            plist = plistlib.loads(plist_xml.encode("utf-8"))

            assert plist["EnvironmentVariables"] == {}


class TestInstallUninstall:
    """Test launchd service installation and removal."""

    def test_install_writes_plist_files(self, tmp_path: Path) -> None:
        """Install should write plist files to LaunchAgents."""
        cfg = Config()
        cfg.schedule.digest_hour = 22
        cfg.schedule.digest_minute = 0

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=True)

        # dry_run writes nothing at all
        agents_dir = tmp_path / "Library" / "LaunchAgents"
        assert not (agents_dir / f"{WORKER_LABEL}.plist").exists()
        assert not agents_dir.exists()

    def test_install_creates_directories(self, tmp_path: Path) -> None:
        """Install should create LaunchAgents and log directories."""
        cfg = Config()

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        agents_dir = tmp_path / "Library" / "LaunchAgents"
        assert agents_dir.exists()

        worker_plist = agents_dir / f"{WORKER_LABEL}.plist"
        assert worker_plist.exists()

        digest_plist = agents_dir / f"{DIGEST_LABEL}.plist"
        assert digest_plist.exists()

    def test_install_returns_plist_paths(self, tmp_path: Path) -> None:
        """Install should return list of created plist paths."""
        cfg = Config()

        with patch("pathlib.Path.home", return_value=tmp_path):
            paths = install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        assert len(paths) == 2
        assert all(p.exists() for p in paths)

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    @patch("subprocess.run")
    def test_install_runs_launchctl_macos(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Install should run launchctl on macOS."""
        cfg = Config()

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        # Should have called launchctl bootstrap and bootout
        assert mock_run.called

    def test_uninstall_removes_plist_files(self, tmp_path: Path) -> None:
        """Uninstall should remove plist files."""
        cfg = Config()

        with patch("pathlib.Path.home", return_value=tmp_path):
            # First install
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

            # Then uninstall
            uninstall(cfg, dry_run=False)

        agents_dir = tmp_path / "Library" / "LaunchAgents"
        assert not (agents_dir / f"{WORKER_LABEL}.plist").exists()
        assert not (agents_dir / f"{DIGEST_LABEL}.plist").exists()

    def test_install_dry_run_prints_commands(self, tmp_path: Path, capsys) -> None:
        """Dry run should print launchctl commands instead of running them."""
        cfg = Config()

        # Only test on non-macOS to avoid actual launchctl calls
        with patch("sys.platform", "linux"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                install(cfg, vas_bin="/usr/local/bin/vas", dry_run=True)

        captured = capsys.readouterr()
        # Should have printed launchctl commands
        assert "launchctl" in captured.out or "launchctl" in captured.err

    def test_uninstall_dry_run_prints_commands(self, tmp_path: Path, capsys) -> None:
        """Dry run should print rm commands instead of deleting."""
        cfg = Config()

        with patch("pathlib.Path.home", return_value=tmp_path):
            install(cfg, vas_bin="/usr/local/bin/vas", dry_run=False)

        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("sys.platform", "linux"):
                uninstall(cfg, dry_run=True)

        captured = capsys.readouterr()
        # Should have printed rm commands
        assert "rm" in captured.out or "rm" in captured.err


class TestInstallDefaults:
    """Test install command defaults."""

    def test_install_default_vas_bin_from_which(self, tmp_path: Path) -> None:
        """Install should use `which vas` as default."""
        cfg = Config()

        with patch("shutil.which", return_value="/usr/local/bin/vas"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                install(cfg, dry_run=False)

        agents_dir = tmp_path / "Library" / "LaunchAgents"
        worker_plist = agents_dir / f"{WORKER_LABEL}.plist"
        plist_text = worker_plist.read_text()

        assert "/usr/local/bin/vas" in plist_text

    def test_install_uses_sys_argv0_fallback(self, tmp_path: Path) -> None:
        """Install should fall back to sys.argv[0] if which fails."""
        cfg = Config()

        with patch("shutil.which", return_value=None):
            with patch("sys.argv", ["/home/user/.venv/bin/vas"]):
                with patch("pathlib.Path.home", return_value=tmp_path):
                    install(cfg, dry_run=False)

        agents_dir = tmp_path / "Library" / "LaunchAgents"
        worker_plist = agents_dir / f"{WORKER_LABEL}.plist"
        plist_text = worker_plist.read_text()

        assert "vas" in plist_text
