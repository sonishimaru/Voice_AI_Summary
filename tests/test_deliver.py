"""Tests for delivery modules: Slack, email, repo, and digest management."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voice_ai_summary.config import Config, DeliverConfig
from voice_ai_summary.db import connect
from voice_ai_summary.deliver import deliver_digest
from voice_ai_summary.deliver.repo import publish_to_repo
from voice_ai_summary.deliver.slack import _chunk_markdown, _markdown_to_mrkdwn, send_slack


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

    def test_deliver_digest_no_enabled_channels(self, tmp_path: Path) -> None:
        """No delivery if no channels enabled."""
        db_path = tmp_path / "test.sqlite3"
        conn = connect(db_path)

        cfg = Config()
        cfg.deliver.slack = False
        cfg.deliver.email = False

        results = deliver_digest(conn, cfg, "2026-09-15", "# Test")

        assert results == {}


class _FakeGitRunner:
    """Records every `git -C <repo> ...` call and returns a canned result per subcommand.

    `results` maps the git subcommand (e.g. "push") to (returncode, stderr). Anything not
    listed succeeds with empty output.
    """

    def __init__(self, results: dict[str, tuple[int, str]] | None = None) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.results = results or {}

    def __call__(self, args: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append((args, kwargs))
        subcommand = args[3]
        returncode, stderr = self.results.get(subcommand, (0, ""))
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


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
        """add, diff --cached, commit, push run in that order, all with -C <repo>."""
        repo = self._make_repo(tmp_path)
        cfg = DeliverConfig(repo=True, repo_path=str(repo))
        runner = _FakeGitRunner(results={"diff": (1, "")})

        publish_to_repo(cfg, "2026-09-15", "# Digest", runner=runner)

        subcommands = [call_args[3] for call_args, _ in runner.calls]
        assert subcommands == ["pull", "add", "diff", "commit", "push"]
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
