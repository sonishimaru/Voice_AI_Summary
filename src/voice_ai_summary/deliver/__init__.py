"""Core delivery logic: send digests to enabled channels with idempotency checks."""

from __future__ import annotations

import sqlite3

from ..config import Config
from ..db import utcnow_iso
from .email import send_email
from .slack import send_slack


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
        Dict mapping channel name to status: "ok", "skipped", or "error: <reason>".
        Missing secrets are treated as errors with descriptive messages.
    """
    results: dict[str, str] = {}

    # Determine which channels to process
    to_deliver = channels or []
    if not to_deliver:
        if cfg.deliver.slack:
            to_deliver.append("slack")
        if cfg.deliver.email:
            to_deliver.append("email")

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
            if channel == "slack":
                _deliver_slack(cfg, markdown)
            elif channel == "email":
                _deliver_email(cfg, markdown)
            else:
                results[channel] = f"error: unknown channel {channel}"
                continue

            # Record success
            conn.execute(
                "INSERT INTO deliveries (scope_key, channel, sent_at, status, detail) "
                "VALUES (?, ?, ?, ?, ?)",
                (day, channel, utcnow_iso(), "ok", None),
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
