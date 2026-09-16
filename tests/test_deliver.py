"""Tests for delivery modules: Slack, email, repo, and digest management."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer

from voice_ai_summary import commands_deliver
from voice_ai_summary import summarize as summarize_mod
from voice_ai_summary import vad as vad_module
from voice_ai_summary.config import Config, DeliverConfig
from voice_ai_summary.db import connect
from voice_ai_summary.deliver import deliver_digest
from voice_ai_summary.deliver.repo import publish_to_repo
from voice_ai_summary.deliver.slack import _chunk_markdown, _markdown_to_mrkdwn, send_slack
from voice_ai_summary.ingest import ingest_file
from voice_ai_summary.summarize import EpisodeSummary
from voice_ai_summary.timeutil import local_day_bounds
from voice_ai_summary.vad import SpeechRegion


class TestSlackMarkdownConversion:
    """Test Markdown to Slack mrkdwn conversion."""

    def test_h1_to_bold(self) -> None:
        """# heading → *heading* (bold)."""
        md = "# Hello World"
        mrkdwn = _markdown_to_mrkdwn(md)
        assert mrkdwn == "*Hello World*"

    def test_h2_to_bold(self) -> None:
        """## heading → *heading* (bold)."""
        md = "## Section"
        mrkdwn = _markdown_to_mrkdwn(md)
        assert mrkdwn == "*Section*"

    def test_bold_conversion(self) -> None:
        """**text** → *text*."""
        md = "This is **bold** text"
        mrkdwn = _markdown_to_mrkdwn(md)
        assert mrkdwn == "This is *bold* text"

    def test_bullets_unchanged(self) -> None:
        """Bullet points should remain unchanged."""
        md = "- item 1\n- item 2"
        mrkdwn = _markdown_to_mrkdwn(md)
        assert mrkdwn == "- item 1\n- item 2"

    def test_multiline_with_headings(self) -> None:
        """Multi-line conversion with mixed content."""
        md = "# Title\n\nSome **bold** text.\n\n## Section\n- bullet"
        mrkdwn = _markdown_to_mrkdwn(md)
        assert "*Title*" in mrkdwn
        assert "*Section*" in mrkdwn
        assert "*bold*" in mrkdwn
        assert "- bullet" in mrkdwn


class TestSlackChunking:
    """Test markdown chunking on paragraph boundaries."""

    def test_short_text_no_chunk(self) -> None:
        """Short text should remain as single chunk."""
        md = "Hello world"
        chunks = _chunk_markdown(md, max_chars=1000)
        assert chunks == ["Hello world"]

    def test_long_text_chunked(self) -> None:
        """Long text should be split into multiple chunks."""
        md = "\n\n".join([f"Paragraph {i}" for i in range(10)])
        chunks = _chunk_markdown(md, max_chars=50)
        assert len(chunks) > 1
        # Each chunk should be under max_chars
        for chunk in chunks:
            assert len(chunk) <= 60  # Allow some overhead

    def test_preserves_paragraph_boundaries(self) -> None:
        """Chunks should be split on paragraph boundaries."""
        md = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = _chunk_markdown(md, max_chars=30)
        # Each chunk should contain complete paragraphs
        for chunk in chunks:
            assert "Paragraph" in chunk

    def test_single_large_paragraph(self) -> None:
        """Single large paragraph that exceeds max_chars should still be included."""
        md = "A" * 5000
        chunks = _chunk_markdown(md, max_chars=1000)
        # Should have at least one chunk
        assert len(chunks) >= 1
        # Concatenated should recover original
        assert "".join(chunks) == md


class TestSlackSending:
    """Test Slack webhook posting."""

    @patch("httpx.post")
    def test_send_slack_success(self, mock_post: MagicMock) -> None:
        """Successful Slack post should complete without error."""
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_post.return_value = mock_resp

        send_slack("https://hooks.slack.com/services/...", "# Hello\n\n**bold**")

        # Should have called httpx.post
        assert mock_post.called
        call_args = mock_post.call_args
        # Should post JSON with text key
        assert call_args.kwargs["json"]["text"]
        assert "*Hello*" in call_args.kwargs["json"]["text"]

    @patch("httpx.post")
    def test_send_slack_failure(self, mock_post: MagicMock) -> None:
        """Failed Slack post should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.is_success = False
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="401"):
            send_slack("https://hooks.slack.com/services/...", "Hello")

    @patch("httpx.post")
    def test_send_slack_chunks(self, mock_post: MagicMock) -> None:
        """Large messages should be sent in chunks."""
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_post.return_value = mock_resp

        # Create a message larger than default chunk size (3500 chars per chunk)
        large_md = "\n\n".join([f"Paragraph {i}: " + "x" * 100 for i in range(100)])
        send_slack("https://hooks.slack.com/services/...", large_md, timeout=30.0)

        # Should have posted multiple times
        assert mock_post.call_count > 1


class TestEmailSending:
    """Test email delivery via SMTP."""

    def test_send_email_with_mock_smtp(self) -> None:
        """Email should be sent with correct headers and body."""
        from voice_ai_summary.deliver.email import send_email

        mock_smtp = MagicMock()
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        cfg = DeliverConfig(
            email=True,
            email_from="sender@example.com",
            email_to="recipient@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user@example.com",
        )

        send_email(
            cfg,
            "password123",
            "Test Subject",
            "Test body",
            smtp_factory=mock_smtp,
        )

        # Should have called SMTP with correct host and port
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        # Should have called starttls and login
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("user@example.com", "password123")


class TestDeliverDigest:
    """Test digest delivery with idempotency."""

    def test_deliver_digest_skips_existing(self, tmp_path: Path) -> None:
        """Digest already sent should be skipped."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        # Insert an existing delivery record
        conn.execute(
            "INSERT INTO deliveries (scope_key, channel, sent_at, status, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-09-15", "slack", "2026-09-15T10:00:00Z", "ok", None),
        )
        conn.commit()

        cfg = Config()
        cfg.deliver.slack = True

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["slack"])

        assert results["slack"] == "skipped"

    def test_deliver_digest_force_resend(self, tmp_path: Path) -> None:
        """Force flag should resend even if already delivered."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        # Insert an existing delivery record
        conn.execute(
            "INSERT INTO deliveries (scope_key, channel, sent_at, status, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            ("2026-09-15", "slack", "2026-09-15T10:00:00Z", "ok", None),
        )
        conn.commit()

        cfg = Config()
        cfg.deliver.slack = True

        with patch("voice_ai_summary.deliver.send_slack"):
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                results = deliver_digest(
                    conn, cfg, "2026-09-15", "# Test", channels=["slack"], force=True
                )

        # Should have attempted resend
        assert results["slack"] == "ok"

    def test_deliver_digest_missing_slack_webhook(self, tmp_path: Path) -> None:
        """Missing Slack webhook should produce error status."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True
        # No VAS_SLACK_WEBHOOK_URL set

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["slack"])

        assert "error" in results["slack"]
        assert "VAS_SLACK_WEBHOOK_URL" in results["slack"]

    def test_deliver_digest_missing_email_password(self, tmp_path: Path) -> None:
        """Missing SMTP password should produce error status."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.email = True
        # No VAS_SMTP_PASSWORD set

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["email"])

        assert "error" in results["email"]
        assert "VAS_SMTP_PASSWORD" in results["email"]

    def test_deliver_digest_records_success(self, tmp_path: Path) -> None:
        """Successful delivery should be recorded in database."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True

        with patch("voice_ai_summary.deliver.send_slack"):
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["slack"])

        # Check that delivery was recorded
        row = conn.execute(
            "SELECT status, detail FROM deliveries WHERE scope_key = ? AND channel = ?",
            ("2026-09-15", "slack"),
        ).fetchone()
        assert row["status"] == "ok"
        assert row["detail"] is None

    def test_deliver_digest_records_error(self, tmp_path: Path) -> None:
        """Failed delivery should be recorded with error detail."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True

        with patch(
            "voice_ai_summary.deliver.send_slack",
            side_effect=RuntimeError("Network error"),
        ):
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["slack"])

        # Check that error was recorded
        row = conn.execute(
            "SELECT status, detail FROM deliveries WHERE scope_key = ? AND channel = ?",
            ("2026-09-15", "slack"),
        ).fetchone()
        assert row["status"] == "error"
        assert "Network error" in row["detail"]

    def test_deliver_digest_skips_no_data_digest_on_every_channel(self, tmp_path: Path) -> None:
        """A no-data placeholder digest must not reach any channel - and especially not
        the repo channel, where it would overwrite a real, already-mirrored file."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True
        cfg.deliver.repo = True
        cfg.deliver.repo_path = str(tmp_path / "repo")

        no_data_markdown = "# 2026-09-16 の記録\n\n記録なし\n"

        with (
            patch("voice_ai_summary.deliver.send_slack") as mock_slack,
            patch("voice_ai_summary.deliver.publish_to_repo") as mock_publish,
        ):
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                results = deliver_digest(
                    conn, cfg, "2026-09-16", no_data_markdown, channels=["slack", "repo"]
                )

        assert results == {
            "slack": "skipped: no-data digest",
            "repo": "skipped: no-data digest",
        }
        mock_slack.assert_not_called()
        mock_publish.assert_not_called()

        # Nothing was recorded as delivered either.
        rows = conn.execute("SELECT * FROM deliveries WHERE scope_key = '2026-09-16'").fetchall()
        assert rows == []

    def test_deliver_digest_skips_no_data_digest_even_with_force(self, tmp_path: Path) -> None:
        """force=True still must not deliver a no-data placeholder."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.repo = True
        cfg.deliver.repo_path = str(tmp_path / "repo")

        no_data_markdown = "# 2026-09-16 の記録\n\n記録なし\n"

        with patch("voice_ai_summary.deliver.publish_to_repo") as mock_publish:
            results = deliver_digest(
                conn, cfg, "2026-09-16", no_data_markdown, channels=["repo"], force=True
            )

        assert results == {"repo": "skipped: no-data digest"}
        mock_publish.assert_not_called()

    def test_deliver_digest_normal_digest_delivers_as_before(self, tmp_path: Path) -> None:
        """Regression check: a real digest still delivers exactly as it did before the
        no-data guard was added."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True

        real_markdown = "# 2026-09-16 の記録\n\n## ハイライト\n\n- 本物のダイジェスト\n"

        with patch("voice_ai_summary.deliver.send_slack") as mock_slack:
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                results = deliver_digest(conn, cfg, "2026-09-16", real_markdown, channels=["slack"])

        assert results == {"slack": "ok"}
        mock_slack.assert_called_once()

        row = conn.execute(
            "SELECT status FROM deliveries WHERE scope_key = ? AND channel = ?",
            ("2026-09-16", "slack"),
        ).fetchone()
        assert row["status"] == "ok"

    def test_deliver_digest_no_enabled_channels(self, tmp_path: Path) -> None:
        """No delivery if no channels enabled."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = False
        cfg.deliver.email = False
        cfg.deliver.notify = False  # on by default: the local-only channel

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test")

        assert results == {}

    def test_deliver_digest_skips_no_data_digest_with_incompleteness_banner(
        self, tmp_path: Path
    ) -> None:
        """`_insert_incomplete_banner` rewrites a no-data placeholder's Markdown before
        `deliver_digest` ever sees it - the placeholder must still be recognised as
        having no data to deliver, and in particular the repo channel must never
        receive it (it would overwrite a real, already-mirrored digest file)."""
        from voice_ai_summary.commands_deliver import _insert_incomplete_banner

        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = True
        cfg.deliver.repo = True
        cfg.deliver.repo_path = str(tmp_path / "repo")

        no_data_markdown = "# 2026-09-16 の記録\n\n記録なし\n"
        banner_markdown = _insert_incomplete_banner(no_data_markdown, missing=3)
        # Sanity check: this really is the rewritten shape, not the raw placeholder.
        assert banner_markdown != no_data_markdown
        assert summarize_mod.is_no_data_digest(banner_markdown)

        with (
            patch("voice_ai_summary.deliver.send_slack") as mock_slack,
            patch("voice_ai_summary.deliver.publish_to_repo") as mock_publish,
        ):
            with patch.dict("os.environ", {"VAS_SLACK_WEBHOOK_URL": "https://example.com"}):
                results = deliver_digest(
                    conn, cfg, "2026-09-16", banner_markdown, channels=["slack", "repo"]
                )

        assert results == {
            "slack": "skipped: no-data digest",
            "repo": "skipped: no-data digest",
        }
        mock_slack.assert_not_called()
        mock_publish.assert_not_called()


class _FakeGitRunner:
    """Records every `git -C <repo> ...` call and returns a canned result per subcommand.

    `results` maps the git subcommand (e.g. "push") to (returncode, stderr). Anything not
    listed succeeds with empty output. `stdouts` maps a subcommand to its stdout; a
    `rev-parse` call defaults to "main" (a plain, non-detached branch name) so tests that
    don't care about branch resolution don't have to configure it.
    """

    def __init__(
        self,
        results: dict[str, tuple[int, str]] | None = None,
        stdouts: dict[str, str] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.results = results or {}
        self.stdouts = stdouts or {}

    def __call__(self, args: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append((args, kwargs))
        subcommand = args[3]
        returncode, stderr = self.results.get(subcommand, (0, ""))
        default_stdout = "main" if subcommand == "rev-parse" else ""
        stdout = self.stdouts.get(subcommand, default_stdout)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class TestRepoDelivery:
    """Test committing and pushing the digest to a local git checkout."""

    def _make_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        return repo

    def test_writes_digest_file(self, tmp_path: Path) -> None:
        """Digest markdown is written under <repo>/<subdir>/<day>.md."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo), repo_subdir="digests")
        runner = _FakeGitRunner(results={"diff": (1, "")})

        path = publish_to_repo(cfg, "2026-09-15", "# Digest content", runner=runner)

        assert path == "digests/2026-09-15.md"
        written = repo / "digests" / "2026-09-15.md"
        assert written.read_text(encoding="utf-8") == "# Digest content"

    def test_add_diff_commit_push_order(self, tmp_path: Path) -> None:
        """rev-parse (branch resolution), pull, add, diff --cached, commit, push run in
        that order, all with -C <repo>."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"diff": (1, "")})

        publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert subcommands == ["rev-parse", "pull", "add", "diff", "commit", "push"]
        for call_args, _ in runner.calls:
            assert call_args[0] == "git"
            assert call_args[1] == "-C"
            assert call_args[2] == str(repo)

    def test_repo_branch_checks_out_and_pushes_to_branch(self, tmp_path: Path) -> None:
        """repo_branch set → checkout runs and push targets HEAD:<branch>."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo), repo_branch="main")
        runner = _FakeGitRunner(results={"diff": (1, "")})

        publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert subcommands[0] == "checkout"
        assert runner.calls[0][0][4] == "main"
        push_call = next(call_args for call_args, _ in runner.calls if call_args[3] == "push")
        assert push_call[-1] == "HEAD:main"

    def test_no_diff_skips_commit(self, tmp_path: Path) -> None:
        """diff --cached exits 0 (no change) → no commit, no exception, path still returned."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"diff": (0, "")})

        path = publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        assert path == "digests/2026-09-15.md"
        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert "commit" not in subcommands
        assert "push" not in subcommands

    def test_failing_push_raises_with_stderr(self, tmp_path: Path) -> None:
        """A failing push raises RuntimeError containing the captured stderr."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"diff": (1, ""), "push": (1, "remote: permission denied")})

        with pytest.raises(RuntimeError, match="permission denied"):
            publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

    def test_failing_pull_does_not_raise(self, tmp_path: Path) -> None:
        """A failing pull (offline) logs a warning but the rest still runs."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"pull": (1, "network unreachable"), "diff": (1, "")})

        path = publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        assert path == "digests/2026-09-15.md"
        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert "commit" in subcommands
        assert "push" in subcommands

    def test_no_branch_configured_pulls_and_pushes_the_same_resolved_branch(
        self, tmp_path: Path
    ) -> None:
        """With `repo_branch` empty, the checkout's actual current branch (as reported
        by `git rev-parse --abbrev-ref HEAD`, here a non-default branch) is resolved
        once and used as both the pull target and the push target - not the literal
        "HEAD", which `git pull` would instead resolve against the *remote's* default
        branch."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(
            results={"diff": (1, "")}, stdouts={"rev-parse": "feature/my-branch"}
        )

        publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        pull_call = next(call_args for call_args, _ in runner.calls if call_args[3] == "pull")
        push_call = next(call_args for call_args, _ in runner.calls if call_args[3] == "push")
        assert pull_call[-1] == "feature/my-branch"
        assert push_call[-1] == "HEAD:feature/my-branch"

    def test_detached_head_skips_pull_but_still_pushes(self, tmp_path: Path) -> None:
        """A detached checkout (rev-parse reports "HEAD" itself) has no local branch
        name to pull into - guessing one would risk fast-forwarding onto an unrelated
        remote ref, so the pull is skipped entirely rather than attempted with a
        guessed ref. The rest of the flow (commit/push) still runs."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"diff": (1, "")}, stdouts={"rev-parse": "HEAD"})

        publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert "pull" not in subcommands
        assert "commit" in subcommands
        assert "push" in subcommands

    def test_missing_repo_path_raises(self) -> None:
        """Empty repo_path raises a RuntimeError naming the problem."""
        cfg = DeliverConfig(repo=True, repo_path="")
        runner = _FakeGitRunner()

        with pytest.raises(RuntimeError, match="repo_path"):
            publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

    def test_repo_path_without_git_raises(self, tmp_path: Path) -> None:
        """A directory that exists but has no .git raises a RuntimeError."""
        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        cfg = DeliverConfig(repo=True, repo_path=str(plain_dir))
        runner = _FakeGitRunner()

        with pytest.raises(RuntimeError, match="not a git repository"):
            publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

    def test_deliver_digest_repo_records_success_and_is_idempotent(self, tmp_path: Path) -> None:
        """deliver_digest with repo enabled records an ok row and skips on resend."""
        repo = self._make_repo(tmp_path)
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.repo = True
        cfg.deliver.repo_path = str(repo)

        runner = _FakeGitRunner(results={"diff": (1, "")})
        with patch("voice_ai_summary.deliver.repo.subprocess.run", runner):
            results = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["repo"])

        assert results["repo"] == "ok"
        row = conn.execute(
            "SELECT status, detail FROM deliveries WHERE scope_key = ? AND channel = ?",
            ("2026-09-15", "repo"),
        ).fetchone()
        assert row["status"] == "ok"
        assert row["detail"] == "digests/2026-09-15.md"

        # Second call is idempotent: skipped, no further git calls.
        call_count_before = len(runner.calls)
        results_again = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["repo"])
        assert results_again["repo"] == "skipped"
        assert len(runner.calls) == call_count_before

    def test_deliver_digest_repo_missing_path_is_error(self, tmp_path: Path) -> None:
        """deliver_digest with repo enabled but no repo_path returns an error string."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.repo = True
        cfg.deliver.repo_path = ""

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test", channels=["repo"])

        assert results["repo"].startswith("error:")
        assert "repo_path" in results["repo"]


class TestNotifyDelivery:
    """macOS notification channel: local-only, nothing leaves the machine."""

    def test_headline_prefers_the_first_highlight_bullet(self) -> None:
        from voice_ai_summary.deliver.notify import headline

        md = "# 2026-09-15 の記録\n\n## ハイライト\n\n- **安田さん**に見積もりを送付\n- 二件目\n"
        assert headline(md) == "安田さんに見積もりを送付"

    def test_headline_falls_back_and_truncates(self) -> None:
        from voice_ai_summary.deliver.notify import headline

        assert headline("# 見出しだけ\n") == "本文を確認してください"
        long_md = "## ハイライト\n- " + "あ" * 300
        assert len(headline(long_md)) == 180 and headline(long_md).endswith("…")

    def test_send_notification_builds_an_escaped_osascript_call(self) -> None:
        from voice_ai_summary.deliver.notify import send_notification

        calls: list[list[str]] = []

        def runner(args, **kwargs):
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        send_notification(
            "2026-09-15", '## ハイライト\n- "引用" を含む行', path="/tmp/x.md", runner=runner
        )

        assert calls[0][0] == "osascript"
        script = calls[0][2]
        assert '\\"引用\\"' in script
        assert 'with title "2026-09-15 のサマリ"' in script
        assert "/tmp/x.md" in script

    def test_send_notification_reports_failure(self) -> None:
        from voice_ai_summary.deliver.notify import send_notification

        def failing(args, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        with pytest.raises(RuntimeError, match="boom"):
            send_notification("2026-09-15", "## ハイライト\n- x", runner=failing)

        def missing(args, **kwargs):
            raise FileNotFoundError

        with pytest.raises(RuntimeError, match="macOS"):
            send_notification("2026-09-15", "## ハイライト\n- x", runner=missing)


def test_notify_is_the_default_channel(tmp_path: Path) -> None:
    """A fresh config delivers the digest to the notification centre and nowhere else."""
    conn = connect(tmp_path / "t.sqlite3")
    cfg = Config()
    calls: list[list[str]] = []

    with patch(
        "voice_ai_summary.deliver.notify.subprocess.run",
        lambda args, **kw: (
            calls.append(args) or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    ):
        results = deliver_digest(conn, cfg, "2026-09-15", "## ハイライト\n- 打ち合わせ")

    assert results == {"notify": "ok"}
    assert calls and calls[0][0] == "osascript"


class TestDigestCatchUp:
    """`vas digest`: transcribe a pending backlog first (bounded by a wall-clock
    budget), then mark an incomplete day in the digest Markdown, stdout, and the
    notification alike - see `commands_deliver._catch_up_and_count_missing`."""

    _EPISODE_SUMMARY = EpisodeSummary(
        title="テスト会話", kind_guess="solo", summary_ja="テスト要約です。"
    )

    @staticmethod
    def _day_markdown(day: str) -> str:
        return f"# {day} の記録\n\n## ハイライト\n\n- テストのハイライト\n"

    def _stub_summarize(self, monkeypatch: pytest.MonkeyPatch, day: str) -> None:
        """Stub the two network calls `run_day` makes, so no anthropic client is needed."""
        monkeypatch.setattr(summarize_mod, "_call_map", lambda *a, **k: self._EPISODE_SUMMARY)
        monkeypatch.setattr(summarize_mod, "_call_reduce", lambda *a, **k: self._day_markdown(day))
        monkeypatch.setattr(summarize_mod.correct, "correct_day", lambda *a, **k: 0)
        monkeypatch.setattr(summarize_mod, "make_client", lambda cfg: object())

    @staticmethod
    def _write_wav(path: Path, seconds: float = 1.0) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * int(seconds * 16000))

    def _ingest_pending(
        self, cfg: Config, conn: object, started_at_utc: str, name: str, seconds: float = 1.0
    ) -> int:
        """A real, still-pending recording inside the day (VAD is faked by the caller)."""
        src = cfg.paths.inbox / f"mac_mic_dev1_{name}.wav"
        self._write_wav(src, seconds=seconds)
        rec_id = ingest_file(conn, cfg, src, started_at_utc=started_at_utc, move=True)
        assert rec_id is not None
        conn.commit()
        return rec_id

    def _insert_errored(self, conn: object, started_at_utc: str, sha: str) -> None:
        """A permanently-errored recording: catch-up cannot fix it, so it stays missing."""
        conn.execute(
            """
            INSERT INTO recordings(
                source, device_id, started_at_utc, tz_offset, duration_ms,
                sha256, storage_path, original_name, ingested_at, error
            ) VALUES ('mac_mic', 'dev1', ?, '+00:00', 1000, ?, 'store/errored.wav',
                      'errored.wav', ?, 'decode failed')
            """,
            (started_at_utc, sha, started_at_utc),
        )
        conn.commit()

    def test_catch_up_transcribes_pending_and_flags_the_remaining_backlog(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, capsys: object
    ) -> None:
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        start_utc, _end_utc = local_day_bounds(day, cfg.summarize.timezone)
        monkeypatch.setattr(
            vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
        )
        self._ingest_pending(cfg, conn, start_utc, "20260915T000000Z")
        self._insert_errored(conn, start_utc, "e" * 64)
        self._stub_summarize(monkeypatch, day)

        commands_deliver.digest(day=day, deliver=False, force=False, channel=None)
        out = capsys.readouterr().out  # type: ignore[attr-defined]

        # The pending recording was transcribed by the catch-up pass...
        pending_left = conn.execute(
            "SELECT COUNT(*) FROM recordings WHERE processed_at IS NULL AND error IS NULL"
        ).fetchone()[0]
        assert pending_left == 0
        # ...but the errored one is still there, so the day is reported as incomplete -
        # in the stdout note and (via the embedded banner) in the digest Markdown too.
        assert "不完全" in out
        assert "1" in out
        from voice_ai_summary.deliver.notify import _INCOMPLETE_PREFIX

        assert _INCOMPLETE_PREFIX in out

    def test_incompleteness_reaches_the_notification_too(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A day that has a real digest (not a no-data placeholder) but is still
        incomplete surfaces the banner in the notification headline too."""
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        start_utc, _end_utc = local_day_bounds(day, cfg.summarize.timezone)
        monkeypatch.setattr(
            vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
        )
        # A real, transcribable recording so the day has actual content (not a no-data
        # placeholder, which the delivery guard must never send anywhere - including
        # notify) - plus an errored one that catch-up cannot fix, so the day still
        # reports as incomplete.
        self._ingest_pending(cfg, conn, start_utc, "20260915T000000Z")
        self._insert_errored(conn, start_utc, "e" * 64)
        self._stub_summarize(monkeypatch, day)

        calls: list[list[str]] = []
        with patch(
            "voice_ai_summary.deliver.notify.subprocess.run",
            lambda args, **kw: (
                calls.append(args) or SimpleNamespace(returncode=0, stdout="", stderr="")
            ),
        ):
            commands_deliver.digest(day=day, deliver=True, force=False, channel=["notify"])

        assert calls, "expected a notification to be sent"
        script = calls[0][2]
        assert "不完全" in script
        assert "1" in script

    def test_budget_is_respected_and_stops_early(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, capsys: object
    ) -> None:
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        monkeypatch.setattr(commands_deliver, "_CATCHUP_BATCH", 1)
        day = "2026-09-15"
        start_utc, _end_utc = local_day_bounds(day, cfg.summarize.timezone)
        monkeypatch.setattr(
            vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
        )
        self._ingest_pending(cfg, conn, start_utc, "a_20260915T000000Z", seconds=1.0)
        self._ingest_pending(cfg, conn, start_utc, "b_20260915T000000Z", seconds=1.5)
        self._stub_summarize(monkeypatch, day)

        # Clock reads: once for the deadline, once to allow the first (one-recording)
        # batch, then a big jump so the loop's second elapsed-check sees the budget as
        # spent - re-checked between batches, exactly as the real clock would be.
        clock_values = iter([0.0, 0.0, 1_000_000.0])
        monkeypatch.setattr(commands_deliver.time, "monotonic", lambda: next(clock_values))

        commands_deliver.digest(day=day, deliver=False, force=False, channel=None)
        out = capsys.readouterr().out  # type: ignore[attr-defined]

        pending_left = conn.execute(
            "SELECT COUNT(*) FROM recordings WHERE processed_at IS NULL AND error IS NULL"
        ).fetchone()[0]
        assert pending_left == 1  # only the first batch ran before the budget ran out
        assert "不完全" in out

    def test_budget_of_zero_skips_catch_up_entirely(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, capsys: object
    ) -> None:
        cfg, conn = vas
        cfg.schedule.digest_catchup_budget_s = 0
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        start_utc, _end_utc = local_day_bounds(day, cfg.summarize.timezone)
        monkeypatch.setattr(
            vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
        )
        self._ingest_pending(cfg, conn, start_utc, "20260915T000000Z")
        self._stub_summarize(monkeypatch, day)

        commands_deliver.digest(day=day, deliver=False, force=False, channel=None)
        out = capsys.readouterr().out  # type: ignore[attr-defined]

        row = conn.execute("SELECT processed_at FROM recordings").fetchone()
        assert row["processed_at"] is None  # catch-up never ran
        assert "不完全" in out

    def test_clean_day_output_is_unchanged(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch, capsys: object
    ) -> None:
        """No pending/errored recordings -> byte-identical to today's (bannerless) output."""
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        self._stub_summarize(monkeypatch, day)

        commands_deliver.digest(day=day, deliver=False, force=False, channel=None)
        out = capsys.readouterr().out  # type: ignore[attr-defined]

        assert out == f"# {day} の記録\n\n記録なし\n\n"
        from voice_ai_summary.deliver.notify import _INCOMPLETE_PREFIX

        assert _INCOMPLETE_PREFIX not in out

    def test_catch_up_only_transcribes_the_digest_days_recordings(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Catch-up must bound `process_pending` to the digest day (via `within=`) -
        otherwise `process_pending`'s global oldest-first order can spend the whole
        `digest_catchup_budget_s` transcribing an unrelated, older day's backlog while
        the day actually being summarized stays untranscribed."""
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        other_day = "2026-09-10"
        start_utc, _end_utc = local_day_bounds(day, cfg.summarize.timezone)
        other_start_utc, _ = local_day_bounds(other_day, cfg.summarize.timezone)
        monkeypatch.setattr(
            vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
        )
        # An older, unrelated day's pending recording: globally-oldest-first (no `within`
        # bound) would pick this up ahead of the digest day's own recording. Different
        # durations keep the two files' content (and thus dedup sha256) distinct.
        other_id = self._ingest_pending(cfg, conn, other_start_utc, "other_day", seconds=1.0)
        same_id = self._ingest_pending(cfg, conn, start_utc, "digest_day", seconds=1.5)
        self._stub_summarize(monkeypatch, day)

        commands_deliver.digest(day=day, deliver=False, force=False, channel=None)

        def _is_processed(rec_id: int) -> bool:
            row = conn.execute(
                "SELECT processed_at FROM recordings WHERE id = ?", (rec_id,)
            ).fetchone()
            return row["processed_at"] is not None

        assert _is_processed(same_id), "the digest day's own recording should be transcribed"
        assert not _is_processed(other_id), (
            "catch-up must not spend its budget on a different day's backlog"
        )


class TestDigestDeliverChannelGuard:
    """`vas digest --deliver`'s "no delivery channels enabled" guard must agree with the
    set of channels `deliver.deliver_digest` actually considers enabled (derived from
    `deliver.enabled_channels`), not a separately hand-maintained list - see
    `commands_deliver.digest`."""

    _EPISODE_SUMMARY = EpisodeSummary(
        title="テスト会話", kind_guess="solo", summary_ja="テスト要約です。"
    )

    def _stub_summarize(self, monkeypatch: pytest.MonkeyPatch, day: str) -> None:
        markdown = f"# {day} の記録\n\n## ハイライト\n\n- テストのハイライト\n"
        monkeypatch.setattr(summarize_mod, "_call_map", lambda *a, **k: self._EPISODE_SUMMARY)
        monkeypatch.setattr(summarize_mod, "_call_reduce", lambda *a, **k: markdown)
        monkeypatch.setattr(summarize_mod.correct, "correct_day", lambda *a, **k: 0)
        monkeypatch.setattr(summarize_mod, "make_client", lambda cfg: object())

    def test_notify_only_reaches_deliver_digest(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh config has only `notify` enabled (its default is True, every other
        channel defaults to False). `vas digest --deliver` must still reach
        `deliver_digest` instead of exiting 1 on the "no channels enabled" guard."""
        cfg, conn = vas
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        assert cfg.deliver.notify
        assert not (cfg.deliver.slack or cfg.deliver.email or cfg.deliver.repo)
        day = "2026-09-15"
        self._stub_summarize(monkeypatch, day)

        with patch("voice_ai_summary.deliver.deliver_digest") as mock_deliver:
            mock_deliver.return_value = {"notify": "ok"}
            commands_deliver.digest(day=day, deliver=True, force=False, channel=None)

        mock_deliver.assert_called_once()

    def test_genuinely_no_channels_still_exits(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With every channel (including `notify`) disabled, the guard still exits 1
        before `deliver_digest` is ever called."""
        cfg, conn = vas
        cfg.deliver.notify = False
        monkeypatch.setattr(commands_deliver, "load_config", lambda: cfg)
        day = "2026-09-15"
        self._stub_summarize(monkeypatch, day)

        with patch("voice_ai_summary.deliver.deliver_digest") as mock_deliver:
            with pytest.raises(typer.Exit):
                commands_deliver.digest(day=day, deliver=True, force=False, channel=None)

        mock_deliver.assert_not_called()
