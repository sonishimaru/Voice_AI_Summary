"""Core delivery logic: send digests to enabled channels with idempotency checks."""

from __future__ import annotations

import sqlite3

from ..config import Config
from ..db import utcnow_iso
from ..summarize import is_no_data_digest
from .email import send_email
from .notify import send_notification
from .repo import publish_to_repo
from .slack import send_slack


def enabled_channels(cfg: Config) -> list[str]:
    """Which channels `deliver_digest` sends to when no explicit `channels` list is
    given - every channel enabled in `cfg.deliver`, in the same order `deliver_digest`
    builds `to_deliver` below.

    Callers that need to know up front whether *any* channel would be attempted (e.g.
    `commands_deliver`'s "no delivery channels enabled" guard) must derive that from
    this function rather than re-listing the channels themselves - a hand-maintained
    second list drifts out of sync with this one (notably: it used to omit `notify`,
    the channel that is enabled by default, so `vas digest --deliver` exited before
    `deliver_digest` ever ran).
    """
    channels: list[str] = []
    if cfg.deliver.slack:
        channels.append("slack")
    if cfg.deliver.email:
        channels.append("email")
    if cfg.deliver.repo:
        channels.append("repo")
    if cfg.deliver.notify:
        channels.append("notify")
    return channels


def deliver_digest(
    conn: sqlite3.Connection,
    cfg: Config,
    day: str,
    markdown: str,
    *,
    channels: list[str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Send digest to enabled channels with idempotency.

    Args:
        conn: Database connection.
        cfg: Configuration.
        day: YYYY-MM-DD date string.
        markdown: Digest markdown content.
        channels: Optional list of channel names to limit delivery to (all enabled if None).
        force: Skip idempotency checks and resend.

    Returns:
        Dict mapping channel name to status: "ok", "skipped", "skipped: no-data digest",
        or "error: <reason>". Missing secrets are treated as errors with descriptive
        messages. A `markdown` that is the "no data" placeholder (see
        `summarize.is_no_data_digest`) is never delivered to any channel, regardless of
        force - there's nothing worth sending, and sending it to the repo channel would
        overwrite a real digest already mirrored there.
    """
    results: dict[str, str] = {}

    # Determine which channels to process
    to_deliver = list(channels) if channels else enabled_channels(cfg)

    if is_no_data_digest(markdown):
        # A "no data" placeholder is never useful to deliver, and through the repo
        # channel it's actively harmful: `publish_to_repo` would overwrite a real,
        # already-mirrored digest file with an empty one. Skip every channel up front,
        # before any channel-specific network/subprocess call is made, and say why -
        # same idea as the per-channel "skipped"/"error: <reason>" statuses below, just
        # decided once for the whole digest instead of per channel.
        for channel in to_deliver:
            results[channel] = "skipped: no-data digest"
        return results

    for channel in to_deliver:
        # Check if already delivered
        if not force:
            row = conn.execute(
                "SELECT status FROM deliveries WHERE scope_key = ? AND channel = ? "
                "AND status = 'ok'",
                (day, channel),
            ).fetchone()
            if row:
                results[channel] = "skipped"
                continue

        # Attempt delivery
        try:
            detail: str | None = None
            if channel == "slack":
                _deliver_slack(cfg, markdown)
            elif channel == "email":
                _deliver_email(cfg, markdown)
            elif channel == "repo":
                detail = _deliver_repo(cfg, day, markdown)
            elif channel == "notify":
                send_notification(day, markdown, path=str(cfg.paths.digests / f"{day}.md"))
            else:
                results[channel] = f"error: unknown channel {channel}"
                continue

            # Record success
            conn.execute(
                "INSERT INTO deliveries (scope_key, channel, sent_at, status, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (day, channel, utcnow_iso(), "ok", detail),
            )
            results[channel] = "ok"
        except Exception as e:
            error_msg = str(e)
            # Record failure
            conn.execute(
                "INSERT INTO deliveries (scope_key, channel, sent_at, status, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (day, channel, utcnow_iso(), "error", error_msg),
            )
            results[channel] = f"error: {error_msg}"

    conn.commit()
    return results


def _deliver_slack(cfg: Config, markdown: str) -> None:
    """Send to Slack. Raises RuntimeError if webhook URL is missing."""
    webhook_url = cfg.slack_webhook_url
    if not webhook_url:
        raise RuntimeError(
            "Slack webhook URL not configured. Set VAS_SLACK_WEBHOOK_URL environment variable."
        )
    send_slack(webhook_url, markdown)


def _deliver_email(cfg: Config, markdown: str) -> None:
    """Send email. Raises RuntimeError if required config is missing."""
    password = cfg.smtp_password
    if not password:
        raise RuntimeError(
            "SMTP password not configured. Set VAS_SMTP_PASSWORD environment variable."
        )
    if not cfg.deliver.email_from:
        raise RuntimeError("email_from not configured in [deliver] section.")
    if not cfg.deliver.email_to:
        raise RuntimeError("email_to not configured in [deliver] section.")

    subject = "Daily Summary"  # Subject line for emails
    send_email(cfg.deliver, password, subject, markdown)


def _deliver_repo(cfg: Config, day: str, markdown: str) -> str:
    """Commit and push to the digest repo. Returns the committed relative path."""
    return publish_to_repo(cfg.deliver, day, markdown)
